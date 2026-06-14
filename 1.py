from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage,AIMessage
import json
app=FastAPI(title="AI Legal Assistant")#搭建了一个网站
#BaseModel自动检查类型
class ChatRequest(BaseModel):
    session_id:str
    #用来区分是来自哪个对话
    message:str

@app.post("/legal/chat")
#异步函数async
#req是参数名，类型为chatrequest
async def legal_chat(req:ChatRequest):
    session=get_session(req:sesion_id)

    history=[]
    for turn in session["history"]:
        history.append(HumanMessage(content=turn["human"]))
        history.appeng(AIMessage(content=turn["ai"]))
    
    if session["case_summary"]:
        case_summary=json.dumps(session["case_summary"],ensure_ascii=False)#保留中文不转义
    else:
        case_summary=""

    result=await run_legal_chat(req.message,history,case_summary)
    answer=result["output"]

    #更新历史,weisha shi session["history"]
    session["history"].append({"human":req.message,"ai":answer})

    #