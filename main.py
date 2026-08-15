"""FastAPI 入口 — AI 法律助手"""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from agent.legal_agent import run_legal_agent, run_legal_agent_stream
from config import HOST, LOCAL_ORIGINS, PORT
from logger import get_logger
from memory.case_memory import get_session, init_db, save_exchange, update_case_summary
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

@app.post("/legal/chat")
async def legal_chat(req: ChatRequest):
    logger.debug(f"[DEBUG] /legal/chat skip_planner={req.skip_planner} msg={req.message[:30]}")
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.perf_counter()
    logger.info(f"[PERF] trace={request_id} stage=request status=start route=chat")

    try:
        session = get_session(req.session_id)

        history = []
        for turn in session["history"]:
            history.append(HumanMessage(content=turn["human"]))
            history.append(AIMessage(content=turn["ai"]))

        case_summary = (
            json.dumps(session["case_summary"], ensure_ascii=False)
            if session["case_summary"]
            else ""
        )

        result = await run_legal_agent(
            req.message,
            history,
            case_summary,
            request_id=request_id,
            skip_planner=req.skip_planner,
        )
        answer = result["output"]

        save_exchange(req.session_id, req.message, answer)

        exchange = f"用户：{req.message}\n助手：{answer}"
        try:
            # 同步更新案情摘要（to_thread 避免阻塞事件循环），
            # 返回给调用方的是本轮更新后的最新摘要，而不是陈旧值。
            updated_summary = await asyncio.to_thread(
                update_case_summary,
                req.session_id,
                exchange,
                request_id,
            )
        except Exception as e:
            logger.error(
                f"[ERROR] trace={request_id} stage=summary "
                f"error={type(e).__name__}: {e}"
            )
            updated_summary = session["case_summary"]

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
            exc_info=True,  # 附带完整堆栈，替代原 traceback.print_exc()
        )
        logger.info(
            f"[PERF] trace={request_id} stage=request_total "
            f"duration_ms={duration_ms:.0f} status=error route=chat "
            f"error_type={type(e).__name__}"
        )
        return {
            "answer": f"服务器内部错误: {type(e).__name__}",
            "session_id": req.session_id,
            "tools_used": [],
            "case_summary": {},
            "error": str(e),
        }


@app.post("/legal/chat/stream")
async def legal_chat_stream(req: ChatRequest):
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.perf_counter()
    logger.info(f"[PERF] trace={request_id} stage=request status=start route=stream")
    session = get_session(req.session_id)

    history = []

    for turn in session["history"]:
        history.append(HumanMessage(content=turn["human"]))
        history.append(AIMessage(content=turn["ai"]))

    case_summary = (
        json.dumps(session["case_summary"], ensure_ascii=False)
        if session["case_summary"]
        else ""
    )

    async def _event_generator_body():
        full_answer = ""

        async for event in run_legal_agent_stream(
            req.message,
            history,
            case_summary,
            request_id=request_id,
            skip_planner=req.skip_planner,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event["type"] in ("token", "planner_question"):
                full_answer += event["text"]

        # 流结束：先持久化本轮对话，再同步更新案情摘要，
        # 并把最新摘要作为 case_summary 事件推给前端（侧边栏实时刷新）。
        # 摘要更新失败不影响已完成的回答，只记日志。
        if full_answer:
            save_exchange(req.session_id, req.message, full_answer)

            exchange = f"用户:{req.message}\n助手:{full_answer}"
            try:
                updated_summary = await asyncio.to_thread(
                    update_case_summary,
                    req.session_id,
                    exchange,
                    request_id,
                    True,
                )
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "case_summary", "data": updated_summary},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            except Exception as e:
                logger.error(
                    f"[ERROR] trace={request_id} stage=summary "
                    f"error={type(e).__name__}: {e}"
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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
