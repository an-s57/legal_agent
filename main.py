"""FastAPI 入口 — AI 法律助手"""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from agent.legal_agent import run_legal_agent, run_legal_agent_stream
from memory.case_memory import get_session, init_db, save_exchange, update_case_summary
from rag.retriever import preload_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库并预加载模型。"""
    init_db()
    preload_reranker()
    yield


app = FastAPI(title="AI Legal Assistant", version="1.0.0", lifespan=lifespan)

# 当前项目用于本地演示；只允许本机网页访问 API。
LOCAL_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

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
    print(f"[OK] Static files mounted: {FRONTEND_DIST / 'assets'}")


class ChatRequest(BaseModel):
    session_id: str
    message: str

QUICK_REPLIES = {
    "你好": "你好！我是法律智能助手。你可以告诉我发生了什么、何时发生、造成了什么后果，以及希望获得什么帮助。",
    "您好": "你好！我是法律智能助手。你可以告诉我发生了什么、何时发生、造成了什么后果，以及希望获得什么帮助。",
    "嗨": "你好！我是法律智能助手。你可以告诉我发生了什么、何时发生、造成了什么后果，以及希望获得什么帮助。",
    "在吗": "我在。你可以描述一下遇到的法律问题，我会协助你梳理相关信息。",
    "谢谢": "不客气！如果还有法律问题，随时可以继续告诉我。",
    "你是谁": "我是一个基于法律知识库和联网检索构建的法律智能助手。",
    "你能做什么": "我可以协助你梳理法律问题、检索相关法律材料，并提示还需要补充哪些关键信息。",
}


def get_quick_reply(message: str):
    normalized = message.strip().rstrip("。！？!?~～")
    return QUICK_REPLIES.get(normalized)
#添加快速回复表

@app.post("/legal/chat")
async def legal_chat(req: ChatRequest):
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.perf_counter()
    print(
        f"[PERF] trace={request_id} stage=request status=start route=chat",
        flush=True,
    )
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
    )
    answer = result["output"]

    save_exchange(req.session_id, req.message, answer)

    exchange = f"用户：{req.message}\n助手：{answer}"
    updated_summary = update_case_summary(
        req.session_id,
        exchange,
        request_id=request_id,
    )

    tools_used = [step[0] for step in result.get("intermediate_steps", [])]

    duration_ms = (time.perf_counter() - request_started_at) * 1000
    print(
        f"[PERF] trace={request_id} stage=request_total "
        f"duration_ms={duration_ms:.0f} status=ok route=chat",
        flush=True,
    )

    return {
        "answer": answer,
        "session_id": req.session_id,
        "tools_used": tools_used,
        "case_summary": updated_summary,
    }


@app.post("/legal/chat/stream")
async def legal_chat_stream(req: ChatRequest,background_tasks:BackgroundTasks):
    request_id = uuid.uuid4().hex[:8]
    request_started_at = time.perf_counter()
    print(
        f"[PERF] trace={request_id} stage=request status=start route=stream",
        flush=True,
    )
    session = get_session(req.session_id)
    quick_reply=get_quick_reply(req.message)
    if quick_reply:
        save_exchange(req.session_id, req.message, quick_reply)

        duration_ms = (time.perf_counter() - request_started_at) * 1000
        print(
            f"[PERF] trace={request_id} stage=request_total "
            f"duration_ms={duration_ms:.0f} status=ok route=quick_reply",
            flush=True,
        )

        async def quick_reply_generator():#异步生成器对象
            token_event={"type":"token","text":quick_reply}
            done_event={"type":"done","tools_used":[]}

            yield f"data: {json.dumps(token_event,ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(done_event,ensure_ascii=False)}\n\n"

        return StreamingResponse(
            quick_reply_generator(),
            media_type="text/event-stream"
        )

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
        full_answer=""

        async for event in run_legal_agent_stream(
            req.message,
            history,
            case_summary,
            request_id=request_id,
        ):
            yield f"data: {json.dumps(event,ensure_ascii=False)}\n\n"

            if event["type"] in ("token","planner_question"):
                full_answer+=event["text"]
        if full_answer:
            save_exchange(req.session_id, req.message, full_answer)

            exchange=f"用户:{req.message}\n助手:{full_answer}"
            background_tasks.add_task(
                update_case_summary,
                req.session_id,
                exchange,
                request_id,
                True,
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
            raise
        finally:
            duration_ms = (time.perf_counter() - request_started_at) * 1000
            error_suffix = f" error_type={error_type}" if error_type else ""
            print(
                f"[PERF] trace={request_id} stage=request_total "
                f"duration_ms={duration_ms:.0f} status={status} route=stream"
                f"{error_suffix}",
                flush=True,
            )
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        background=background_tasks,
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

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
