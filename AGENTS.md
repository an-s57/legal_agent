# AGENTS.md

## 运行前提

- **Ollama** 必须在本机 `http://127.0.0.1:11434` 运行，且已拉取 `nomic-embed-text` 模型（`ollama pull nomic-embed-text`）。
- **虚拟环境：** 运行任何命令前先激活 `.venv`。
- **API key：** 在 `.env` 中设置 `DEEPSEEK_API_KEY`；如需使用联网主搜索，再设置 `ANYSEARCH_API_KEY`（该文件已被 gitignore）。LLM 用的是 DeepSeek V4 Flash，通过 LangChain 的 `ChatOpenAI` 封装调用 DeepSeek API（`api.deepseek.com/v1`）；AnySearch 不可用时联网工具会回退到 `ddgs`。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 构建 FAISS 向量库（服务器启动前必须先跑一次）
python build_vectorstore.py

# 构建前端（首次或前端代码变更后）
cd frontend && npm install && npm run build && cd ..

# 启动服务器
python main.py
# 或者：uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

接口：
- `POST /legal/chat` — 非流式问答
- `POST /legal/chat/stream` — SSE 流式问答（前端使用此接口）
- `GET /legal/session/{id}` — 查询会话（仅本地演示；当前未做鉴权）
- `GET /health` — 健康检查

## 项目架构

```
main.py （FastAPI 入口，端口 8000）
  ├─ agent/legal_agent.py     — LangGraph 智能体（Planner + ReAct + SSE 流式）
  │    ├─ tools/legal_tools.py  — legal_rag_search + web_legal_search（AnySearch 主搜索 + ddgs 兜底）
  │    └─ rag/retriever.py      — FAISS 检索 + CrossEncoder Reranker + 自定义 OllamaEmbeddings
  ├─ memory/case_memory.py    — SQLite 会话/消息持久化 + LLM 增量案情摘要
  └─ frontend/                — React + TypeScript + Tailwind CSS
```

Agent 状态图流程：`START → planner（信息完整性检查）→ llm（ReAct 决策）⇄ tools（条件跳转）→ END`。

Planner 节点先判断用户消息是否包含四个关键维度（事件描述、时间、损失/后果、诉求）。缺失则生成自然追问，信息完整后才进入 ReAct 检索+回答流程。

## 注意事项

- **`rag/vectorstore/` 被 gitignore 了** — 首次运行前必须执行 `python build_vectorstore.py` 建库。
- **自定义 `OllamaEmbeddings`**（`rag/retriever.py`）直接调 Ollama 旧版 `/api/embeddings` 接口，并设置 `trust_env=False` 避免被系统代理拦截。**不要**替换成 `langchain_ollama.OllamaEmbeddings`，它与当前安装的 Ollama 客户端版本不兼容。
- **FAISS 加载用了 `allow_dangerous_deserialization=True`**（`rag/retriever.py`），因为 FAISS 索引是用 pickle 序列化的，必须加这个参数才能加载。
- **会话已持久化到 `data/legal_agent.db`**（`memory/case_memory.py`）：`sessions` 保存会话和案情摘要，`messages` 保存多轮消息；服务重启后可按 `session_id` 恢复。前端把当前会话 ID 保存到浏览器 `localStorage`，刷新后恢复这一个会话（侧栏历史列表暂不跨刷新保存）。不要提交本地数据库文件或真实会话数据。
- **`recursion_limit=12`**（`agent/legal_agent.py`）限制了 LangGraph 最多 ~5 轮工具调用，防止 ReAct 循环失控。
- **Reranker 在 FastAPI lifespan 中预加载**（`main.py`），避免首次请求等待 5s+。
- **`1.py`、`2.py`、`3.py`** 是学习/草稿文件，直接忽略。真正的入口是 `main.py`。
- **项目暂无 pytest 等自动化单元测试**；已有检索参数评测和联网搜索对比脚本，不能把它们等同于完整回归测试。
- `.env` 已被 gitignore，但里面包含真实 API key，**千万不要提交**。
