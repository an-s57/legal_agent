# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from agent.legal_agent import run_legal_agent
from memory.case_memory import get_session, update_case_summary
import json

app = FastAPI(title="AI Legal Assistant")

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/legal/chat")
async def legal_chat(req: ChatRequest):
    session = get_session(req.session_id)
    
    # 把历史转成LangChain消息格式
    history = []
    for turn in session["history"]:
        history.append(HumanMessage(content=turn["human"]))
        history.append(AIMessage(content=turn["ai"]))
    
    # 拿案情摘要
    case_summary = json.dumps(
        session["case_summary"], ensure_ascii=False
    ) if session["case_summary"] else ""
    
    # 跑Agent
    result = await run_legal_agent(req.message, history, case_summary)
    answer = result["output"]
    
    # 更新历史
    session["history"].append({"human": req.message, "ai": answer})
    
    # 异步更新案情摘要（不阻塞返回）
    exchange = f"用户：{req.message}\n助手：{answer}"
    updated_summary = update_case_summary(req.session_id, exchange)
    
    # 暴露工具调用链，面试展示用
    tools_used = []
    for step in result.get("intermediate_steps", []):
        tools_used.append(step[0])
    
    return {
        "answer": answer,
        "session_id": req.session_id,
        "tools_used": tools_used,          # 能看到这次用了哪些工具
        "case_summary": updated_summary    # 能看到案情摘要实时更新
    }

@app.get("/legal/session/{session_id}")
async def get_session_info(session_id: str):
    """查看某个会话的完整状态，调试用"""
    return get_session(session_id)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)