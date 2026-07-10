# AI 法律助手

基于 **LangGraph + RAG + FastAPI** 的智能法律问答系统。用户输入法律问题后，Agent 自主决策调用 RAG 法条检索或联网搜索，整合后给出带来源标注的回答，同时维护多轮会话记忆和案情摘要。

## 架构

```
用户提问
  │
  ▼
FastAPI (/legal/chat)
  │
  ▼
LangGraph ReAct Agent ──→ should_continue?
  │                           │
  ├─ legal_rag_search         ├─ 有 tool_calls → 执行工具
  │   (FAISS 向量检索)        │
  │                           └─ 无 tool_calls → 输出最终回答
  ├─ web_legal_search
  │   (DuckDuckGo 联网搜索)
  │
  ▼
更新会话历史 + LLM 案情摘要
  │
  ▼
返回回答 + 工具调用链 + 案情摘要
```

## 项目结构

```
legal_agent/
├── main.py                  # FastAPI 入口
├── agent/
│   └── legal_agent.py       # LangGraph ReAct 智能体（手写 StateGraph）
├── tools/
│   └── legal_tools.py       # legal_rag_search + web_legal_search
├── rag/
│   └── retriever.py         # FAISS 检索 + 自定义 OllamaEmbeddings
├── memory/
│   └── case_memory.py       # 会话管理 + LLM 案情摘要
├── build_vectorstore.py     # 构建向量库（首次运行执行一次）
├── legal_pdfs/              # 法律 PDF 源文件
└── docs/                    # 学习笔记与周报
```

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | GLM-4-flash（智谱 AI） |
| Agent 框架 | LangGraph（手写 StateGraph，非 AgentExecutor） |
| 向量库 | FAISS（本地） |
| Embedding | nomic-embed-text（Ollama 本地服务） |
| Web 搜索 | DuckDuckGo |
| Web 框架 | FastAPI + Uvicorn |

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/an-s57/legal_agent.git
cd legal_agent

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

在项目根目录创建 `.env` 文件：

```
GLM_API_KEY=你的智谱API密钥
```

### 3. 启动 Ollama Embedding 服务

```bash
# 安装 Ollama 后拉取嵌入模型
ollama pull nomic-embed-text
```

### 4. 构建向量库

将法律 PDF 文件放入 `legal_pdfs/` 目录，然后执行：

```bash
python build_vectorstore.py
```

### 5. 启动服务

```bash
python main.py
# 或 uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API 接口

### POST /legal/chat

发送法律问题，获取智能回答。

```json
// Request
{
  "session_id": "session-001",
  "message": "拖欠工资三个月，我该怎么维权？"
}

// Response
{
  "answer": "根据《劳动合同法》...",
  "session_id": "session-001",
  "tools_used": ["legal_rag_search"],
  "case_summary": {
    "case_type": "劳动纠纷",
    "key_facts": ["拖欠工资三个月"],
    "user_claim": "维权"
  }
}
```

### GET /legal/session/{session_id}

查询指定会话的历史记录和案情摘要。

### GET /health

健康检查。

## 核心设计

### 为什么手写 LangGraph StateGraph？

`create_agent()` 是黑盒，手写 StateGraph 可以完全控制 Agent 的状态流转：每个节点（`llm`、`tools`）和边（条件跳转 `should_continue`）都是显式定义的，便于调试和扩展。

### 为什么自定义 OllamaEmbeddings？

`langchain_ollama.OllamaEmbeddings` 与当前安装的 Ollama 客户端版本不兼容。自定义类直接用 `httpx` 调用 Ollama 旧版 `/api/embeddings` 接口，并设置 `trust_env=False` 避免被系统代理拦截。

### 两层记忆机制

- **短期记忆**：保存原始对话历史（`history`），用于上下文连续性
- **长期记忆**：LLM 增量提取案情摘要（`case_summary`），压缩为结构化 JSON（案件类型/关键事实/用户诉求），注入下一轮对话的 SystemMessage

## License

MIT
