# AI 法律助手

基于 **LangGraph + RAG + FastAPI** 的智能法律问答系统。Agent 先通过 Planner 节点判断用户信息是否完整（缺失则追问），信息完备后自主决策调用 RAG 法条检索或联网搜索，整合后给出带来源标注的回答，同时维护多轮会话记忆和案情摘要。

## 预览

> TODO：运行前端后截图/GIF 放这里 — 面试官只看图，这是最重要的！

## 架构

flowchart TD
    A["👤 用户提问"] --> B["FastAPI<br/>POST /legal/chat/stream"]
    B --> C["加载会话上下文<br/>history + case_summary"]
    C --> D["🧠 Planner 节点<br/>信息完整性检查"]

    D --> E{"四个维度<br/>是否齐备？"}
    E -->|"❌ 缺失 → 追问用户"| F["返回追问<br/>等待用户补充"]
    E -->|"✅ 完整 → 放行"| G["🤖 LLM 决策<br/>ReAct 循环"]

    G --> H{"有 tool_calls？"}
    H -->|"有"| I["🔧 执行工具"]
    I --> I1["legal_rag_search<br/>📚 FAISS 粗召回 20 条<br/>🎯 Reranker 精排 Top 5"]
    I --> I2["web_legal_search<br/>🌐 DuckDuckGo 联网搜索"]
    I1 --> G
    I2 --> G
    H -->|"无 → 输出回答"| J["📤 SSE 流式输出<br/>逐 token 推送前端"]

    J --> K["💾 更新会话<br/>history + case_summary"]
    K --> L["✅ 返回完整回答<br/>+ 工具调用链 + 案情摘要"]

## 项目结构

```
legal_agent/
├── main.py                  # FastAPI 入口
├── frontend/                # React + TypeScript + Tailwind 前端
│   ├── src/
│   │   ├── components/      # ChatArea, Sidebar, InputBox 等
│   │   └── App.tsx
│   └── dist/                # 构建产物（npm run build）
├── agent/
│   └── legal_agent.py       # LangGraph 智能体（Planner + ReAct）
├── tools/
│   └── legal_tools.py       # legal_rag_search + web_legal_search
├── rag/
│   └── retriever.py         # FAISS 检索 + Reranker + 自定义 OllamaEmbeddings
├── memory/
│   └── case_memory.py       # 会话管理 + LLM 案情摘要
├── evaluation/              # 评测系统（检索质量 + 工具选择）
├── build_vectorstore.py     # 构建向量库（首次运行执行一次）
├── legal_pdfs/              # 法律 PDF 源文件
└── docs/                    # 学习笔记与周报
```

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | GLM-4.7（智谱 AI） |
| Agent 框架 | LangGraph（手写 StateGraph，含 Planner + ReAct） |
| 向量库 | FAISS（本地） |
| Embedding | nomic-embed-text（Ollama 本地服务） |
| Reranker | BAAI/bge-reranker-base（CrossEncoder） |
| Web 搜索 | DuckDuckGo |
| 后端 | FastAPI + Uvicorn |
| 前端 | React + TypeScript + Tailwind CSS |

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

### 5. 构建前端（可选，首次需要）

```bash
cd frontend
npm install
npm run build      # 生成 dist/
cd ..
```

### 6. 启动服务

```bash
python main.py
# 或 uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000

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
    "event_description": "公司拖欠三个月工资",
    "user_claim": "维权追回工资"
  }
}
```

### GET /legal/session/{session_id}

查询指定会话的历史记录和案情摘要。

### GET /health

健康检查。

## 核心设计

### Planner 节点：先收集信息，再回答

在传统 ReAct 之前增加 Planner 节点，判断用户输入是否包含四个关键维度（事件描述、时间、损失、诉求）。缺失则生成自然的追问，信息完备后才进入检索+回答流程。使用同一个 LLM 调用完成分析和追问生成，零额外成本。

### 为什么手写 LangGraph StateGraph？

`create_agent()` 是黑盒，手写 StateGraph 可以完全控制 Agent 的状态流转：每个节点（`planner`、`llm`、`tools`）和边（条件跳转 `should_continue`）都是显式定义的，便于调试和扩展。

### 为什么自定义 OllamaEmbeddings？

`langchain_ollama.OllamaEmbeddings` 与当前安装的 Ollama 客户端版本不兼容。自定义类直接用 `httpx` 调用 Ollama 旧版 `/api/embeddings` 接口，并设置 `trust_env=False` 避免被系统代理拦截。

### 两层记忆机制

- **短期记忆**：保存原始对话历史（`history`），用于上下文连续性
- **长期记忆**：LLM 增量提取案情摘要（`case_summary`），压缩为结构化 JSON（案件类型/关键事实/用户诉求），注入下一轮对话的 SystemMessage

### Reranker 二次排序

FAISS 粗召回 20 条后，用 CrossEncoder（`bge-reranker-base`）对 query-doc 对重新打分，取 top 5。比纯 FAISS 的余弦相似度更精准，本地 CPU 可运行，无需额外 API 费用。

## License

MIT
