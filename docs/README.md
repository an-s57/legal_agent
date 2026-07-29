# 法律 AI 助手 — 学习笔记

围绕法律智能问答助手持续整理的 Agent、RAG 与后端学习笔记。

**项目仓库：** [an-s57/legal_agent](https://github.com/an-s57/legal_agent)

---

## 项目简介

> 用户输入法律问题 → FastAPI 接收 → SQLite 恢复会话 → Planner 判断信息是否齐全 → Agent 决定调用本地 RAG 或联网搜索 → 整合回答并持久化会话

| 模块 | 用途 |
|------|------|
| `agent/legal_agent.py` | LangGraph StateGraph：Planner + ReAct 决策循环 |
| `tools/legal_tools.py` | 两个工具：RAG 检索法条 / 联网查新规（AnySearch 主搜索 + ddgs 兜底） |
| `rag/retriever.py` | FAISS 向量库，自定义 Ollama 嵌入 |
| `memory/case_memory.py` | SQLite 会话/消息持久化 + LLM 增量案情摘要 |
| `main.py` | FastAPI 入口，串联全流程 |

---

## 技术栈

- **LLM：** GLM-4.7（Agent 推理）+ nomic-embed-text（文本转向量）
- **框架：** LangGraph + LangChain + FastAPI
- **向量库：** FAISS（本地）
- **嵌入服务：** Ollama（本地 127.0.0.1:11434）
- **搜索：** AnySearch（主搜索）+ ddgs（失败兜底）
- **会话：** SQLite

---

## 每周小记

- [2026-05-29 复盘](weekly/2026-05-29.md) — 向量库建库成功、Agent 工具调用跑通
- [2026-06-14 复盘](weekly/2026-06-14.md) — 六级后复盘，学习路径总结
- [2026-07-27 复盘](weekly/2026-07-27.md) — SQLite 持久化、检索参数调优与联网搜索服务对比
