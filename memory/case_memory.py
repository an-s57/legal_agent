"""会话管理 — SQLite 会话/消息持久化 + LLM 增量案情摘要。"""
import json
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


#数据库
DATA_DIR=Path(__file__).resolve().parent.parent/"data"
DB_PATH=DATA_DIR/"legal_agent.db"
#数据库路径

def init_db()->None:
    DATA_DIR.mkdir(parents=True,exist_ok=True)

    conn=sqlite3.connect(DB_PATH)
    try:
        #execute向数据库执行SQL操作
        conn.execute("PRAGMA foreign_keys=ON")#消息必须属于某个会话的关系
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY,
            case_summary TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN('human','ai')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id)
                REFERENCES sessions(session_id)
                ON DELETE CASCADE  
            )
""")#如果未来删除一个会话，那么属于它的所有消息也一起删除。#避免插入脏数据

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
            ON messages(session_id,id)
""")#建立索引
        conn.commit()
    finally:
        conn.close()#无论如何都执行，否则数据库文件会被一直占用
    #创建了数据库和2张空表



def save_exchange(session_id:str,human:str,ai:str)->None:
     """保存一轮用户与助手的对话。"""
     conn=sqlite3.connect(DB_PATH)
     try:
         conn.execute("PRAGMA foreign_keys=ON")
         conn.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id)
            VALUES(?)
            """,
            (session_id,),#单元素元组
            )#?是SQL占位符    确保这个会话存在
         conn.execute(
             """
            INSERT INTO messages (session_id,role,content)
            VALUES(?,?,?)
            """,#这三列的具体值稍后由 Python 传进来。
            (session_id,"human",human),
         )
         #存入用户的这句话
         conn.execute(
             """
            INSERT INTO messages(session_id,role,content)
            VALUES(?,?,?)
            """,
            (session_id,"ai",ai)
         )
         #存入AI的这句话
         conn.execute(
             """
            UPDATE sessions
            SET updated_at=CURRENT_TIMESTAMP
            WHERE session_id=?
            """,
            (session_id,),
         )#更新会话最后修改时间

         conn.commit()
     finally:
        conn.close()



def save_case_summary(session_id:str,case_summary:dict)->None:
    """把最新案情摘要写入 SQLite"""
    conn=sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions(session_id) 
            VALUES(?)
            """,
            (session_id,),
        )
        conn.execute(
            """
            UPDATE sessions
            SET case_summary=?,updated_at=CURRENT_TIMESTAMP
            WHERE session_id=?
            """,
            (json.dumps(case_summary,ensure_ascii=False),session_id),
        )
        conn.commit()
    finally:
        conn.close()

#从数据库里面取出历史对话和案情摘要再交给agent作为上下文
def load_session_from_db(session_id:str)->dict:
    """从 SQLite 读取一个会话；没有记录时返回空会话。"""
    conn=sqlite3.connect(DB_PATH)
    try:
        summary_row=conn.execute(
            """
            SELECT case_summary
            FROM sessions
            WHERE session_id=?
            """,
            (session_id,),
            ).fetchone()#案情摘要
        message_rows=conn.execute(
            """
            SELECT role,content
            FROM messages
            WHERE session_id=?
            ORDER BY id
            """,
            (session_id,),

        ).fetchall()#历史对话

        case_summary=json.loads(summary_row[0]) if summary_row else {}
        history=[]
        current_human=None
        #拼接
        for role,content in message_rows:
            if role=="human":
                current_human=content
            elif role=="ai" and current_human is not None:
                history.append({"human":current_human,"ai":content})
                current_human=None
        return{"history":history,"case_summary":case_summary}
    finally:
        conn.close()


llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key=os.getenv("GLM_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)


def get_session(session_id: str) -> dict:
    """读取指定会话的持久化数据。"""
    return load_session_from_db(session_id)


def update_case_summary(
    session_id: str,
    new_exchange: str,
    request_id: str = "",
    background: bool = False,
) -> dict:
    """每轮对话后调用，让 LLM 更新案情摘要"""
    started_at = time.perf_counter()
    status = "error"

    try:
        session = get_session(session_id)
        current = json.dumps(session["case_summary"], ensure_ascii=False)

        prompt = f"""请根据以下对话提取或更新案件关键信息，以JSON格式返回，不要包含任何其他内容：

当前摘要：{current}
新对话：{new_exchange}

输出格式：
{{
  "case_type": "案件类型，如劳动纠纷/合同纠纷/刑事案件，未知则为空字符串",
  "event_description": "事件描述：发生了什么事情，未知则为空字符串",
  "event_time": "事情发生时间，未知则为空字符串",
  "damages": "损失或后果：造成了什么损失，未知则为空字符串",
  "user_claim": "用户诉求，未知则为空字符串"
}}"""

        response = llm.invoke(prompt)
        try:
            text = response.content.strip().strip("```json").strip("```").strip()
            updated = json.loads(text)
            save_case_summary(session_id, updated)
            status = "ok"
            return updated
        except Exception:
            status = "parse_fallback"
            return session["case_summary"]
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        trace = request_id or "-"
        print(
            f"[PERF] trace={trace} stage=summary "
            f"duration_ms={duration_ms:.0f} status={status} "
            f"background={str(background).lower()}",
            flush=True,
        )
