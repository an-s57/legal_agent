"""FastAPI 入口 — AI 法律助手"""
import asyncio
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from agent.legal_agent import run_legal_agent, run_legal_agent_stream
from cache.redis_client import get_answer, set_answer, get_intent_decision, get_cache_stats
from config import HOST, LOCAL_ORIGINS, OPS_QA_TOKEN, PORT
from logger import get_logger
from memory.case_memory import get_session, init_db, save_exchange, update_case_summary
from ops_data_qa.mysql_ops_query import ask as ops_ask
from rag.retriever import preload_reranker

logger = get_logger("legal_agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库并预加载模型。"""
    init_db()
    preload_reranker()
    yield


app = FastAPI(title="AI Legal Assistant", version="1.0.0", lifespan=lifespan)

# 当前项目用于本地演示；只允许本机网页访问 API（白名单在 config.py）。
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# React 前端静态文件
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
    logger.info(f"[OK] Static files mounted: {FRONTEND_DIST / 'assets'}")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    skip_planner: bool = False


# ── 两条路由（/legal/chat 与 /legal/chat/stream）共用的会话/缓存/落库逻辑 ──

def _load_conversation(session_id: str) -> dict:
    """加载会话上下文：历史消息、案情摘要、是否"无上下文首问"。

    两条路由共用，保证缓存读写门控与上下文判定语义一致。
    """
    session = get_session(session_id)
    history = []
    for turn in session["history"]:
        history.append(HumanMessage(content=turn["human"]))
        history.append(AIMessage(content=turn["ai"]))
    return {
        "history": history,
        "case_summary_str": (
            json.dumps(session["case_summary"], ensure_ascii=False)
            if session["case_summary"]
            else ""
        ),
        "case_summary": session["case_summary"],
        # 无上下文首问判定：history 空 + 无案情摘要（与 call_planner 的 _ctx_free 同语义）
        "ctx_free": (not session["history"]) and (not session["case_summary"]),
    }


def _lookup_answer_cache(
    session_id: str, message: str, ctx_free: bool, skip_planner: bool
) -> str | None:
    """回答缓存读取（1.1B）：客观知识问题命中则跳过整条 Agent 链路；命中即落库。"""
    if not ctx_free or skip_planner:
        return None
    cached = get_answer(message)
    if cached is None:
        return None
    logger.info(f"[CACHE] answer hit: {message[:40]}")
    save_exchange(session_id, message, cached)
    return cached


def _maybe_store_answer(message: str, answer: str, ctx_free: bool, skip_planner: bool) -> None:
    """回答缓存写入门控（1.1B）：仅"无上下文 + Planner 判定客观知识"才缓存整条回答。

    案情咨询的答案绝不入缓存（防止把个性化答案错发给别人）。
    """
    if not ctx_free or skip_planner or not answer:
        return
    if answer.startswith("服务器内部错误"):
        return
    intent = get_intent_decision(message)
    if intent and intent.get("is_general_knowledge"):
        set_answer(message, answer)


async def _persist_and_summarize(
    session_id: str,
    message: str,
    answer: str,
    request_id: str,
    background: bool = False,
) -> dict | None:
    """保存本轮对话，再同步更新案情摘要（to_thread 避免阻塞事件循环）。

    摘要更新失败不影响已保存的回答：返回 None，由调用方决定兜底行为。
    """
    save_exchange(session_id, message, answer)
    exchange = f"用户：{message}\n助手：{answer}"
    try:
        return await asyncio.to_thread(
            update_case_summary,
            session_id,
            exchange,
            request_id,
            background,
        )
    except Exception as e:
        logger.error(
            f"[ERROR] trace={request_id} stage=summary "
            f"error={type(e).__name__}: {e}"
        )
        return None


@app.post("/legal/chat")
async def legal_chat(req: ChatRequest):
    logger.debug(f"[DEBUG] /legal/chat skip_planner={req.skip_planner} msg={req.message[:30]}")
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.perf_counter()
    logger.info(f"[PERF] trace={request_id} stage=request status=start route=chat")

    try:
        ctx = _load_conversation(req.session_id)

        # ── 回答缓存读取（1.1B）：命中则秒回 ──
        cached = _lookup_answer_cache(req.session_id, req.message, ctx["ctx_free"], req.skip_planner)
        if cached is not None:
            duration_ms = (time.perf_counter() - request_started_at) * 1000
            logger.info(
                f"[PERF] trace={request_id} stage=request_total "
                f"duration_ms={duration_ms:.0f} status=ok route=chat cache=answer_hit"
            )
            return {
                "answer": cached,
                "session_id": req.session_id,
                "tools_used": [],
                "case_summary": ctx["case_summary"],
            }

        result = await run_legal_agent(
            req.message,
            ctx["history"],
            ctx["case_summary_str"],
            request_id=request_id,
            skip_planner=req.skip_planner,
        )
        answer = result["output"]

        # ── 回答缓存写入（1.1B）：门控见 _maybe_store_answer ──
        _maybe_store_answer(req.message, answer, ctx["ctx_free"], req.skip_planner)

        updated_summary = await _persist_and_summarize(
            req.session_id, req.message, answer, request_id
        )
        if updated_summary is None:
            updated_summary = ctx["case_summary"]   # 摘要更新失败 → 返回旧摘要

        tools_used = [step[0] for step in result.get("intermediate_steps", [])]

        duration_ms = (time.perf_counter() - request_started_at) * 1000
        logger.info(
            f"[PERF] trace={request_id} stage=request_total "
            f"duration_ms={duration_ms:.0f} status=ok route=chat"
        )

        return {
            "answer": answer,
            "session_id": req.session_id,
            "tools_used": tools_used,
            "case_summary": updated_summary,
        }

    except Exception as e:
        duration_ms = (time.perf_counter() - request_started_at) * 1000
        logger.error(
            f"[ERROR] trace={request_id} error={type(e).__name__}: {e}",
            exc_info=True,  # 附带完整堆栈
        )
        logger.info(
            f"[PERF] trace={request_id} stage=request_total "
            f"duration_ms={duration_ms:.0f} status=error route=chat "
            f"error_type={type(e).__name__}"
        )
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {type(e).__name__}")


@app.post("/legal/chat/stream")
async def legal_chat_stream(req: ChatRequest):
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.perf_counter()
    logger.info(f"[PERF] trace={request_id} stage=request status=start route=stream")

    # 会话加载放在流开始之前：失败时提前返回错误事件，而不是让异常直接断开连接
    try:
        ctx = _load_conversation(req.session_id)
    except Exception as e:
        duration_ms = (time.perf_counter() - request_started_at) * 1000
        logger.error(
            f"[ERROR] trace={request_id} stage=load_session "
            f"error={type(e).__name__}: {e}",
            exc_info=True,
        )
        logger.info(
            f"[PERF] trace={request_id} stage=request_total "
            f"duration_ms={duration_ms:.0f} status=error route=stream "
            f"error_type={type(e).__name__}"
        )

        async def _load_error_stream():
            _err = {"type": "error", "message": "会话加载失败，请稍后重试"}
            yield f"data: {json.dumps(_err, ensure_ascii=False)}\n\n"

        return StreamingResponse(_load_error_stream(), media_type="text/event-stream")

    async def _event_generator_body():
        full_answer = ""

        # ── 回答缓存读取（1.1B）：命中则把缓存回答分段推成 token，再发 done，跳过整条链路 ──
        cached = _lookup_answer_cache(req.session_id, req.message, ctx["ctx_free"], req.skip_planner)
        if cached is not None:
            for _i in range(0, len(cached), 24):
                _chunk = {"type": "token", "text": cached[_i:_i + 24]}
                yield f"data: {json.dumps(_chunk, ensure_ascii=False)}\n\n"
            _done = {"type": "done", "tools_used": [], "cache": "answer_hit"}
            yield f"data: {json.dumps(_done, ensure_ascii=False)}\n\n"
            return

        async for event in run_legal_agent_stream(
            req.message,
            ctx["history"],
            ctx["case_summary_str"],
            request_id=request_id,
            skip_planner=req.skip_planner,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event["type"] in ("token", "planner_question"):
                full_answer += event["text"]

        # ── 回答缓存写入（1.1B）：门控见 _maybe_store_answer ──
        _maybe_store_answer(req.message, full_answer, ctx["ctx_free"], req.skip_planner)

        # 流结束：先持久化本轮对话，再同步更新案情摘要，
        # 并把最新摘要作为 case_summary 事件推给前端（侧边栏实时刷新）。
        if full_answer:
            updated_summary = await _persist_and_summarize(
                req.session_id, req.message, full_answer, request_id, background=True,
            )
            if updated_summary is not None:
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "case_summary", "data": updated_summary},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

    async def event_generator():
        status = "error"
        error_type = ""

        try:
            async for payload in _event_generator_body():
                yield payload
            status = "ok"
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            yield f'data: {json.dumps({"type": "error", "message": f"请求处理失败: {error_type}"}, ensure_ascii=False)}\n\n'
            raise
        finally:
            duration_ms = (time.perf_counter() - request_started_at) * 1000
            error_suffix = f" error_type={error_type}" if error_type else ""
            logger.info(
                f"[PERF] trace={request_id} stage=request_total "
                f"duration_ms={duration_ms:.0f} status={status} route=stream"
                f"{error_suffix}"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


class OpsQaRequest(BaseModel):
    message: str
    explain: bool = False


@app.post("/ops/qa")
async def ops_qa(req: OpsQaRequest, x_ops_token: str = Header(default="", alias="X-OPS-TOKEN")):
    """运营数据问答（text-to-SQL 查询网关）。

    后台专用接口：请求头必须带 X-OPS-TOKEN（值在 .env 的 OPS_QA_TOKEN，不配置则接口停用）。
    链路：意图拦截 → LLM 生成 SQL → sqlglot 语法树校验 → 只读账号执行 → 结果人话化。
    ask() 内部是同步 LLM + MySQL 调用，丢线程池避免阻塞事件循环。
    """
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.perf_counter()

    if not OPS_QA_TOKEN:
        raise HTTPException(status_code=503, detail="运营问答未启用：请在 .env 配置 OPS_QA_TOKEN")
    if not hmac.compare_digest(x_ops_token, OPS_QA_TOKEN):
        raise HTTPException(status_code=401, detail="X-OPS-TOKEN 校验失败")

    result = await asyncio.to_thread(ops_ask, req.message, req.explain)
    elapsed_ms = round((time.perf_counter() - request_started_at) * 1000)
    result["elapsed_ms"] = elapsed_ms
    result["trace"] = request_id
    logger.info(
        f"[PERF] trace={request_id} stage=ops_qa duration_ms={elapsed_ms} "
        f"allowed={result.get('allowed')} retries={result.get('retries')} "
        f"error={result.get('error')}"
    )
    return result


@app.get("/legal/session/{session_id}")
async def get_session_info(session_id: str):
    return get_session(session_id)


@app.get("/", response_class=HTMLResponse)
async def index():
    index_html = FRONTEND_DIST / "index.html"
    if not index_html.exists():
        return HTMLResponse(
            "<h2>前端尚未构建</h2>"
            "<p>请在项目根目录执行：<code>cd frontend && npm install && npm run build</code></p>",
            status_code=503,
        )
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.get("/legal/cache/stats")
async def cache_stats():
    """只读：三层缓存的命中/未命中/不可用/写入计数 + 命中率。进程重启清零。"""
    return get_cache_stats()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
