"""FastAPI 入口 — AI 法律助手"""
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from agent.legal_agent import run_legal_agent
from memory.case_memory import get_session, update_case_summary

app = FastAPI(title="AI Legal Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


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


@app.get("/legal/session/{session_id}")
async def get_session_info(session_id: str):
    return get_session(session_id)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
