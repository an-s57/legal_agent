# 法律 AI 助手 — 学习笔记

一个 AI 法律助手项目，从零开始学习 Agent + RAG 架构。

**项目仓库：** [an-s57/legal_agent](https://github.com/an-s57/legal_agent)

---

## 项目简介

> 用户输入法律问题 → FastAPI 接收 → Agent 决定调用哪个工具 → RAG 查法条 / 联网搜索 → 整合回答

| 模块 | 用途 |
|------|------|
| `agent/legal_agent.py` | LangChain Agent，ReAct 决策循环 |
| `tools/legal_tools.py` | 两个工具：RAG 检索法条 / 联网查新规 |
| `rag/retriever.py` | FAISS 向量库，自定义 Ollama 嵌入 |
| `memory/case_memory.py` | 会话管理 + LLM 增量案情摘要 |
| `main.py` | FastAPI 入口，串联全流程 |

---

## 技术栈

- **LLM：** GLM-4-flash（Agent 推理）+ nomic-embed-text（文本转向量）
- **框架：** LangChain 1.3 + FastAPI
- **向量库：** FAISS（本地）
- **嵌入服务：** Ollama（本地 127.0.0.1:11434）
- **搜索：** DuckDuckGo

---

## 每周小记

- [2026-05-29 复盘](weekly/2026-05-29.md) — 向量库建库成功、Agent 工具调用跑通
- [2026-06-14 复盘](weekly/2026-06-14.md) — 六级后复盘，学习路径总结
