# memory/case_memory.py
# 使用简单的内存字典，保存会话和案情摘要
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, json

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.getenv("GLM_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 全局会话数据：{session_id: {"history": [], "case_summary": {}}}
sessions: dict = {}

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "case_summary": {}
        }
    return sessions[session_id]

def update_case_summary(session_id: str, new_exchange: str) -> dict:
    """每轮对话后调用，让LLM更新案情摘要"""
    session = get_session(session_id)
    current = json.dumps(session["case_summary"], ensure_ascii=False)

    prompt = f"""请根据以下对话提取或更新案件关键信息，以JSON格式返回，不要包含任何其他内容：

当前摘要：{current}
新对话：{new_exchange}

输出格式：
{{
  "case_type": "案件类型，如劳动纠纷/合同纠纷/刑事案件，未知则为空字符串",
  "key_facts": ["关键事实1", "关键事实2"],
  "user_claim": "用户诉求，未知则为空字符串"
}}"""

    response = llm.invoke(prompt)
    try:
        text = response.content.strip().strip("```json").strip("```").strip()
        updated = json.loads(text)
        session["case_summary"] = updated
        return updated
    except Exception:
        return session["case_summary"]
