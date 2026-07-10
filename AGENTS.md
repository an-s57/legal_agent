# AGENTS.md

## 运行前提

- **Ollama** 必须在本机 `http://127.0.0.1:11434` 运行，且已拉取 `nomic-embed-text` 模型（`ollama pull nomic-embed-text`）。
- **虚拟环境：** 运行任何命令前先激活 `.venv`。
- **API key：** 在 `.env` 中设置 `GLM_API_KEY`（该文件已被 gitignore）。LLM 用的是智谱 GLM-4-flash，通过 LangChain 的 `ChatOpenAI` 封装调用智谱 API（`open.bigmodel.cn`），**不是** OpenAI。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 构建 FAISS 向量库（服务器启动前必须先跑一次）
python build_vectorstore.py

# 启动服务器
python main.py
# 或者：uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

接口：`POST /legal/chat`、`GET /legal/session/{id}`、`GET /health`。

## 项目架构

```
main.py （FastAPI 入口，端口 8000）
  └─ agent/legal_agent.py   — LangGraph ReAct 智能体
       ├─ tools/legal_tools.py  — legal_rag_search + web_legal_search（DuckDuckGo 搜索）
       └─ rag/retriever.py      — FAISS 检索 + 自定义 OllamaEmbeddings
  └─ memory/case_memory.py  — 内存会话字典 + LLM 案情摘要
```

Agent 使用 LangGraph（不是 LangChain 的 AgentExecutor）。状态图流程：`START → llm → tools（条件跳转）→ llm → END`。

## 注意事项

- **`rag/vectorstore/` 被 gitignore 了** — 首次运行前必须执行 `python build_vectorstore.py` 建库。
- **自定义 `OllamaEmbeddings`**（`rag/retriever.py:15`）直接调 Ollama 旧版 `/api/embeddings` 接口，并设置 `trust_env=False` 避免被系统代理拦截。**不要**替换成 `langchain_ollama.OllamaEmbeddings`，它与当前安装的 Ollama 客户端版本不兼容。
- **FAISS 加载用了 `allow_dangerous_deserialization=True`**（`rag/retriever.py:82`），因为 FAISS 索引是用 pickle 序列化的，必须加这个参数才能加载。
- **会话存在内存里**（`memory/case_memory.py:16`）— 服务器重启后所有会话丢失。
- **`1.py`、`2.py`、`3.py`** 是学习/草稿文件，直接忽略。真正的入口是 `main.py`。
- **项目没有测试**。
- `.env` 已被 gitignore，但里面包含真实 API key，**千万不要提交**。
