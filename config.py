"""集中管理所有可调参数/常量 — 单一数据源，改一处全局生效。

原则：
- 只放"会被多处引用"或"需要统一调整"的值；纯一次性/局部常量留在原处。
- 环境变量优先（如 OLLAMA_BASE_URL 由 docker-compose 注入），默认值兜底。
- 目前是纯 Python 模块；以后可升级为 pydantic-settings 从 .env 读取。
"""
import os
from pathlib import Path

# ── Agent / 上下文管理（agent/legal_agent.py）──
MAX_HISTORY_TURNS = 12        # 传给 LLM 的最大对话轮数（1 轮 = 用户 + AI 各一条）
PLANNER_CONTEXT_TURNS = 3     # 传给 Planner 的最近对话轮数（理解多轮上下文）
RECURSION_LIMIT = 12          # LangGraph 递归上限（约 ~5 轮工具调用），防 ReAct 循环失控
MAX_TOOL_ROUNDS = 5           # 单次请求最多工具调用轮次，超过强制 END（防死循环）

# ── 检索（rag/retriever.py + rag/hybrid.py）──
FAISS_DB_PATH = "rag/vectorstore/db_faiss"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")  # docker-compose 会注入
OLLAMA_EMBED_MODEL = "nomic-embed-text"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANK_MAX_LENGTH = 512       # reranker 输入截断长度
RETRIEVAL_K_VECTOR = 40       # 评测定案：k=40 是"最小安全候选池"，不能砍小
RETRIEVAL_K_BM25 = 20
RETRIEVAL_RRF_K = 60
RETRIEVAL_CANDIDATES = 40
RETRIEVAL_TOP_K = 5

# ── 联网搜索（tools/legal_tools.py）──
ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
WEB_SEARCH_SUFFIX = "法律法规 中国"
WEB_SEARCH_TOP_N = 3
WEB_SEARCH_TIMEOUT_SECONDS = 10.0

# ── 记忆（memory/case_memory.py）──
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "legal_agent.db"
MAX_LOAD_MESSAGES = 100       # 最多从 DB 加载最近 100 条消息（50 轮），Agent 层会进一步截断

# ── 服务器（main.py）──
LOCAL_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
HOST = "127.0.0.1"
PORT = 8000
