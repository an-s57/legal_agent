"""集中管理所有可调参数/常量 — 单一数据源，改一处全局生效。

原则：
- 只放"会被多处引用"或"需要统一调整"的值；纯一次性/局部常量留在原处。
- 环境变量优先（如 OLLAMA_BASE_URL 由 docker-compose 注入），默认值兜底。
- 目前是纯 Python 模块；以后可升级为 pydantic-settings 从 .env 读取。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# config 是几乎所有模块的第一个导入：在这里加载 .env，保证不管走服务入口
# 还是命令行入口（ops_data_qa 脚本、评测 runner），环境配置都已就位。
# 不覆盖已有环境变量（override=False），CI 里显式设置的值优先。
load_dotenv()

# ── Agent / 上下文管理（agent/legal_agent.py）──
MAX_HISTORY_TURNS = 12        # 传给 LLM 的最大对话轮数（1 轮 = 用户 + AI 各一条）
PLANNER_CONTEXT_TURNS = 3     # 传给 Planner 的最近对话轮数（理解多轮上下文）
RECURSION_LIMIT = 12          # LangGraph 递归上限（约 ~5 轮工具调用），防 ReAct 循环失控
MAX_TOOL_ROUNDS = 5           # 单次请求最多工具调用轮次，超过强制 END（防死循环）

# ── 检索（rag/retriever.py + rag/hybrid.py）──
FAISS_DB_PATH = "rag/vectorstore/db_faiss"
# 向量库版本号：重建/增量更新向量库后手动 +1（v1→v2）。
# 回答缓存 + 检索缓存的 key 都带它，改版本号 = 让这两类旧缓存全部自动失效。
VECTORSTORE_VERSION = "v1"
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

# ── LLM 模型（llm_client.py）──
# 生成模型：主 Agent 回答 + text-to-SQL 生成。
MAIN_MODEL = os.getenv("LEGAL_AGENT_MODEL", "deepseek-v4-flash")
MAIN_API_BASE = os.getenv("LEGAL_AGENT_API_BASE", "https://api.deepseek.com/v1")
# 判卷模型：与生成模型异构（不同厂商），用于合规复核 Agent + 运营问答的 LLM 意图门。
# 当前用 GLM-4.5-Air：智谱账号里这个模型额度最稳。
# 教训（真实事故）：曾用 GLM-4.7，余额耗尽后每次调用 429 → 意图门 fail-open 静默放行，
# 评测报告看起来只是"没拦住"，其实是"这一层根本没工作"。换模型只改这里或 .env 的
# LEGAL_AGENT_JUDGE_MODEL，代码别处不再出现模型名字面量。
JUDGE_MODEL = os.getenv("LEGAL_AGENT_JUDGE_MODEL", "glm-4.5-air")
JUDGE_API_BASE = os.getenv("LEGAL_AGENT_JUDGE_API_BASE", "https://open.bigmodel.cn/api/paas/v4")

# ── 运营数据问答（ops_data_qa/，text-to-SQL）──
OPS_DB_HOST = os.getenv("OPS_DB_HOST", "127.0.0.1")
OPS_DB_PORT = int(os.getenv("OPS_DB_PORT", "3306"))
OPS_DB_USER = os.getenv("OPS_DB_USER", "query_user")
OPS_DB_PASSWORD = os.getenv("OPS_DB_PASSWORD", "")   # 真实密码放 .env，绝不写进代码
OPS_DB_NAME = os.getenv("OPS_DB_NAME", "ops_demo")
OPS_QA_MAX_ROWS = 200         # 结果行数上限：没写 LIMIT 的查询自动补上
OPS_QA_MAX_RETRY = 2          # SQL 执行报错后喂回 LLM 重试次数
# 后台接口 /ops/qa 的鉴权令牌；不配置 = 接口停用（fail-closed）
OPS_QA_TOKEN = os.getenv("OPS_QA_TOKEN", "")

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
