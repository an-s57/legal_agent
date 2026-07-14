"""FastAPI 入口 — AI 法律助手"""
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/health")
async def health():
    return {"status": "ok"}


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 法律助手</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #f0f2f5; height: 100vh; display: flex; }
  .sidebar { width: 280px; background: #1a1a2e; color: #fff; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .sidebar h1 { font-size: 18px; color: #e0e0e0; }
  .sidebar label { font-size: 13px; color: #999; }
  .sidebar input { background: #16213e; border: 1px solid #30475e; border-radius: 6px; padding: 8px 12px; color: #fff; font-size: 14px; }
  .sidebar input:focus { outline: none; border-color: #4f8cff; }
  .info-card { background: #16213e; border-radius: 8px; padding: 12px; font-size: 12px; color: #aaa; }
  .info-card b { color: #4f8cff; }
  .main { flex: 1; display: flex; flex-direction: column; }
  .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 70%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.6; }
  .msg.user { align-self: flex-end; background: #4f8cff; color: #fff; }
  .msg.bot { align-self: flex-start; background: #fff; color: #333; border: 1px solid #e0e0e0; }
  .msg.bot .tools { margin-top: 8px; font-size: 12px; color: #4f8cff; }
  .input-bar { padding: 16px 24px; background: #fff; border-top: 1px solid #e0e0e0; display: flex; gap: 12px; }
  .input-bar input { flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; font-size: 14px; }
  .input-bar input:focus { outline: none; border-color: #4f8cff; }
  .input-bar button { background: #4f8cff; color: #fff; border: none; border-radius: 8px; padding: 12px 24px; font-size: 14px; cursor: pointer; }
  .input-bar button:hover { background: #3a7bf0; }
  .input-bar button:disabled { background: #ccc; cursor: not-allowed; }
  .typing { color: #999; font-size: 13px; align-self: flex-start; }
</style>
</head>
<body>
<div class="sidebar">
  <h1>AI 法律助手</h1>
  <div>
    <label>会话 ID</label><br>
    <input type="text" id="sessionId" value="" style="width:100%; margin-top:4px;">
  </div>
  <div>
    <button onclick="document.getElementById('sessionId').value='session-'+Date.now(); loadSession();" style="background:#30475e;color:#fff;border:none;border-radius:6px;padding:8px;width:100%;cursor:pointer;">新建会话</button>
  </div>
  <div class="info-card">
    <b>案情摘要</b>
    <div id="caseSummary" style="margin-top:8px;">暂无</div>
  </div>
  <div class="info-card">
    <b>使用说明</b><br>
    1. 新建会话或输入已有 ID<br>
    2. 输入法律问题<br>
    3. Agent 会先判断信息是否完整<br>
    4. 信息不全会追问，完整后检索法条
  </div>
</div>
<div class="main">
  <div class="messages" id="messages">
    <div class="msg bot">你好，我是 AI 法律助手。请描述你遇到的法律问题，我会先了解案情再为你查找相关法条。</div>
  </div>
  <div class="input-bar">
    <input type="text" id="msgInput" placeholder="输入你的问题..." onkeypress="if(event.key==='Enter')sendMsg()">
    <button id="sendBtn" onclick="sendMsg()">发送</button>
  </div>
</div>
<script>
function sendMsg() {
  const input = document.getElementById('msgInput');
  const msg = input.value.trim();
  if (!msg) return;
  const sid = document.getElementById('sessionId').value || ('session-' + Date.now());
  document.getElementById('sessionId').value = sid;

  input.value = '';
  document.getElementById('sendBtn').disabled = true;

  const messages = document.getElementById('messages');
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.textContent = msg;
  messages.appendChild(userDiv);

  const typingDiv = document.createElement('div');
  typingDiv.className = 'typing';
  typingDiv.textContent = '正在思考...';
  messages.appendChild(typingDiv);
  messages.scrollTop = messages.scrollHeight;

  fetch('/legal/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sid, message: msg})
  })
  .then(r => r.json())
  .then(data => {
    typingDiv.remove();
    const botDiv = document.createElement('div');
    botDiv.className = 'msg bot';
    let html = '';
    if (Array.isArray(data.answer)) {
      html = data.answer.map((s, i) => (i+1) + '. ' + s).join('<br>');
    } else {
      html = data.answer.replace(/\n/g, '<br>');
    }
    if (data.tools_used && data.tools_used.length > 0) {
      html += '<div class="tools">使用工具: ' + data.tools_used.join(', ') + '</div>';
    }
    botDiv.innerHTML = html;
    messages.appendChild(botDiv);
    messages.scrollTop = messages.scrollHeight;

    const summary = data.case_summary;
    if (summary && Object.keys(summary).length > 0) {
      let s = '';
      if (summary.case_type) s += '类型: ' + summary.case_type + '<br>';
      if (summary.event_description) s += '事件: ' + summary.event_description + '<br>';
      if (summary.event_time) s += '时间: ' + summary.event_time + '<br>';
      if (summary.damages) s += '损失: ' + summary.damages + '<br>';
      if (summary.user_claim) s += '诉求: ' + summary.user_claim;
      document.getElementById('caseSummary').innerHTML = s || '暂无';
    }
    document.getElementById('sendBtn').disabled = false;
  })
  .catch(err => {
    typingDiv.remove();
    const errDiv = document.createElement('div');
    errDiv.className = 'msg bot';
    errDiv.textContent = '请求失败: ' + err;
    messages.appendChild(errDiv);
    document.getElementById('sendBtn').disabled = false;
  });
}
function loadSession() {
  const sid = document.getElementById('sessionId').value;
  if (!sid) return;
  fetch('/legal/session/' + sid)
    .then(r => r.json())
    .then(data => {
      const messages = document.getElementById('messages');
      messages.innerHTML = '';
      if (data.history && data.history.length > 0) {
        data.history.forEach(turn => {
          const u = document.createElement('div');
          u.className = 'msg user';
          u.textContent = turn.human;
          messages.appendChild(u);
          const b = document.createElement('div');
          b.className = 'msg bot';
          b.innerHTML = Array.isArray(turn.ai) ? turn.ai.join('<br>') : turn.ai.replace(/\n/g,'<br>');
          messages.appendChild(b);
        });
      } else {
        const div = document.createElement('div');
        div.className = 'msg bot';
        div.textContent = '你好，我是 AI 法律助手。请描述你遇到的法律问题。';
        messages.appendChild(div);
      }
      messages.scrollTop = messages.scrollHeight;
    });
}
document.getElementById('sessionId').value = 'session-' + Date.now();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
