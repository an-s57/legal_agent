"""FastAPI 入口 — AI 法律助手"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from agent.legal_agent import run_legal_agent, run_legal_agent_stream
from memory.case_memory import get_session, update_case_summary
from rag.retriever import preload_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载模型，避免首次请求等待"""
    preload_reranker()
    yield


app = FastAPI(title="AI Legal Assistant", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    result = await run_legal_agent(req.message, history, case_summary)
    answer = result["output"]

    session["history"].append({"human": req.message, "ai": answer})

    exchange = f"用户：{req.message}\n助手：{answer}"
    updated_summary = update_case_summary(req.session_id, exchange)

    tools_used = [step[0] for step in result.get("intermediate_steps", [])]

    return {
        "answer": answer,
        "session_id": req.session_id,
        "tools_used": tools_used,
        "case_summary": updated_summary,
    }


@app.post("/legal/chat/stream")
async def legal_chat_stream(req: ChatRequest,background_tasks:BackgroundTasks):
    session = get_session(req.session_id)
    quick_reply=get_quick_reply(req.message)
    if quick_reply:
        session["history"].append({"human":req.message,"ai":quick_reply})

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

    async def event_generator():
        full_answer=""

        async for event in run_legal_agent_stream(req.message,history,case_summary):
            yield f"data: {json.dumps(event,ensure_ascii=False)}\n\n"

            if event["type"] in ("token","planner_question"):
                full_answer+=event["text"]
        if full_answer:
            session["history"].append({"human":req.message,"ai":full_answer})

            exchange=f"用户:{req.message}\n助手:{full_answer}"
            background_tasks.add_task(
                update_case_summary,
                req.session_id,exchange
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
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
