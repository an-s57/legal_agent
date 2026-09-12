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

# 运行离线单元测试（不依赖 Ollama / LLM / 联网）
python -m unittest discover -s tests -v
```

接口：
- `POST /legal/chat` — 非流式问答
- `POST /legal/chat/stream` — SSE 流式问答（前端使用此接口）
- `GET /legal/session/{id}` — 查询会话（仅本地演示；当前未做鉴权）
- `POST /ops/qa` — 运营数据问答（text-to-SQL；请求头 `X-OPS-TOKEN` 必须匹配 `.env` 的 `OPS_QA_TOKEN`，不配置则 503）
- `GET /health` — 健康检查

## 项目架构

```
main.py （FastAPI 入口，端口 8000）
  ├─ agent/legal_agent.py         — LangGraph 智能体（Planner + ReAct + SSE 流式）
  │    ├─ tools/legal_tools.py      — legal_rag_search + web_legal_search（AnySearch 主搜索 + ddgs 兜底）
  │    ├─ rag/retriever.py          — 混合检索：FAISS 向量 + BM25 词面 + RRF 融合 + CrossEncoder Reranker + 自定义 OllamaEmbeddings
  │    ├─ agent/hallucination_guard.py — 幻觉守卫（规则校验：引用存在性 + 覆盖度）
  │    └─ agent/review_agent.py       — 异构判卷（LLM-as-Judge，语义级事实一致性复核；判卷模型见 config.JUDGE_MODEL，当前 GLM-4.5-Air）
  ├─ memory/case_memory.py        — SQLite 会话/消息持久化 + LLM 增量案情摘要
  ├─ ops_data_qa/                 — 运营数据问答（text-to-SQL）：词面拦截 + GLM 意图门 + sqlglot 校验 + 只读执行 + 55 题评测（run_eval.py）
  └─ frontend/                    — React + TypeScript + Tailwind CSS
```

Agent 状态图流程：`START → planner（信息完整性检查）→ llm（ReAct 决策）⇄ tools（条件跳转）→ END`。

注意：`tools` 节点用的是**自定义 `_dedup_tool_node`**（`agent/legal_agent.py`），在原生 ToolNode 之外包了**工具去重 + 轮次计数（MAX_TOOL_ROUNDS）**：相同 `(tool, args_hash)` 只真调一次、后续直接返回缓存结果，防止同一工具反复调用、ReAct 循环失控。

Planner 节点先判断用户消息是否包含四个关键维度（事件描述、时间、损失/后果、诉求）。缺失则生成自然追问，信息完整后才进入 ReAct 检索+回答流程。

## 注意事项

- **`rag/vectorstore/` 被 gitignore 了** — 首次运行前必须执行 `python build_vectorstore.py` 建库。
- **自定义 `OllamaEmbeddings`**（`rag/retriever.py`）直接调 Ollama 旧版 `/api/embeddings` 接口，并设置 `trust_env=False` 避免被系统代理拦截。**不要**替换成 `langchain_ollama.OllamaEmbeddings`，它与当前安装的 Ollama 客户端版本不兼容。
- **FAISS 加载用了 `allow_dangerous_deserialization=True`**（`rag/retriever.py`），因为 FAISS 索引是用 pickle 序列化的，必须加这个参数才能加载。
- **会话已持久化到 `data/legal_agent.db`**（`memory/case_memory.py`）：`sessions` 保存会话和案情摘要，`messages` 保存多轮消息；服务重启后可按 `session_id` 恢复。前端把当前会话 ID 保存到浏览器 `localStorage`，刷新后恢复这一个会话（侧栏历史列表暂不跨刷新保存）。不要提交本地数据库文件或真实会话数据。
- **`recursion_limit=12`**（`agent/legal_agent.py`）限制了 LangGraph 最多 ~5 轮工具调用，防止 ReAct 循环失控。
- **Reranker 在 FastAPI lifespan 中预加载**（`main.py`），避免首次请求等待 5s+。
- **Redis 缓存（三层，均 fail-open，key 做归一化+sha256）：** ① 意图缓存 `intent_cache:`（`cache/redis_client.py`）缓存 Planner 判定，仅对"无上下文首问"生效，值里带 `is_general_knowledge`；② 回答缓存 `answer_cache:{ver}:` 缓存客观知识问题的**完整回答**，命中即跳过整条 Agent 链路秒回（`main.py` 两路由接入；写门控=无上下文+`is_general_knowledge`，案情咨询绝不缓存）；③ 检索结果缓存 `retrieval_cache:{ver}_k{K}_tk{TOPK}:`（`tools/legal_tools.py` 的 `legal_rag_search`）缓存检索产出的法条原文，命中即跳过最慢的 hybrid+rerank（~3s）；缓存的是客观原文故无需语义门控，空结果也缓存。**失效开关统一在 `config.VECTORSTORE_VERSION`**：重建/增量更新向量库后手动 +1（v1→v2），回答缓存与检索缓存 key 同时变、旧缓存自动失效（旧 key 靠 TTL 自然过期）。`RETRIEVAL_CACHE_TTL` 默认 7 天，`CACHE_ENABLED=0` 可整体关闭（CI/离线测试用）。**命中率统计**：`GET /legal/cache/stats` 返回三层缓存 hit/miss/unavailable/write + 命中率（`cache/redis_client.py` 内 `threading.Lock` 保护的内存计数，fail-safe、进程重启清零；Redis 不可用单独记 `unavailable`，不计入 miss）。
- **单元测试：** `tests/` 下 129 个纯内存离线单测（`python -m unittest discover -s tests -v`，毫秒级跑完，CI 自动执行），覆盖 SQLite 会话持久化、混合检索、幻觉守卫、Planner 决策、意图/回答/检索缓存行为、命中率统计与 fail-open、Redis 重连冷却、ops SQL 语法树校验（含旧正则版漏洞回归用例）、LLM 意图门（stub 判卷 + 多轮投票/提前收敛）、**评测比较器容差**（`test_ops_eval_compare.py`：比例↔百分比、舍入精度、整数不设容差）等，LLM 与 Redis 均用 stub/mock 替换。`evaluation/` 与 `ops_data_qa/run_eval.py` 需真实 LLM/数据库，仅开发期手动跑，不等同于回归测试。
- **运营数据问答四道门**（`ops_data_qa/`，纵深防御）：①词面意图拦截（fail-closed）→ ②LLM 意图门（`OPS_INTENT_LLM_ENABLED` 开关，fail-open；判卷模型 GLM，见 `config.JUDGE_MODEL`，`OPS_INTENT_VOTES` 轮多数投票）→ ③sqlglot 语法树校验（单条 SELECT/表白名单/危险函数/自动补 LIMIT；表白名单只有 sessions、messages、answer_ratings）→ ④MySQL 只读账号。数据库连接与令牌全部走 `.env`（`OPS_DB_*`、`OPS_QA_TOKEN`），**代码里不落密码**；`mysql_lab/make_big_table.py` 造数需 `OPS_DB_ADMIN_PASSWORD`（root，仅本地实验）。**注意 fail-open 陷阱**：意图门不可用（如判卷模型余额耗尽 429）时会静默放行，`ask()` 结果里的 `intent_gate` 字段与评测报告里的 `intent_gate` 段就是用来暴露这件事的 —— 报告若显示 `unavailable`，这批数据不能用来证明意图门有效。
- `.env` 已被 gitignore，但里面包含真实 API key，**千万不要提交**。
