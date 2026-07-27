# 法律 RAG 检索参数评测说明

## 这份评测在测什么

`retrieval_k_tuning_50.json` 是一组用于调优检索候选数 `k` 的本地 RAG 评测题，共 50 道。

它只评估一个问题：**在固定的法律语料、Embedding、切分方式和 Reranker 配置下，FAISS 先召回多少条候选文本，才能让目标法律依据稳定进入重排后的结果。**

它不用于评估：

- Agent 是否正确选择联网搜索工具；
- 大模型最终回答是否足够通顺、完整；
- 法律建议在现实案件中的正确性或时效性。

## 语料边界

本题集只使用当前 FAISS 已实际入库的 5 部法律文件：

| 法律文件 | 题数 |
| --- | ---: |
| 《消费者权益保护法》 | 12 |
| 《产品质量法》 | 10 |
| 《反不正当竞争法》 | 10 |
| 《广告法》 | 8 |
| 《食品安全法》 | 10 |

当前向量库共有 110 个文本块。目录中的《电子商务法》PDF 目前无法被 PDFPlumber 正常提取，未进入 FAISS；因此旧题集中的电商法题不参加本轮 `k` 调优。

这意味着：本题集验证的是“项目当前语料快照下的检索能力”，并不等于这些法律文件一定是最新、最适合真实法律咨询的版本。法律语料更新后，应重新核验题目和重新跑基线。

## 每道题的标注

每个 JSON 记录包含：

- `question`：用户式提问；
- `expected_sources`：目标法律文件；
- `gold_evidence.article`：目标条款；
- `gold_evidence.pdf_page`：人工核验用的 PDF 页码（从 1 开始；FAISS 的 `metadata.page` 从 0 开始）；
- `gold_evidence.anchor`：用于人工核验的关键证据提示；
- `difficulty`：检索表达难度，不是法律难度；
- `split`：`development` 用于观察和调参，`validation` 只用于最终确认。

当前 `evaluate.py` 的 `Hit@5` 仍是“最终 Top-5 中是否出现目标来源文件”的基础指标。后续可升级为同时检查 `source + article/page + anchor`，以避免同一部法律中的无关条款也被判为命中。

## 后续调 K 的正确顺序

1. 固定语料、Embedding 模型、切分参数、Reranker 模型和 `top_k=5`。
2. 只改变 `k`，例如依次比较 `k=10 / 15 / 20 / 30`。
3. 先看开发集的命中率与 Rerank 耗时，再用验证集确认结果。
4. 选择“验证集表现接近最好、但候选更少且更快”的最小 `k`，而不是只追求最高分。
5. 只有当 `k` 固定后，再单独评估是否需要修改最终交给 LLM 的 `top_k`。

不要把联网题、Planner 追问题或语料未入库的题混进这一步；它们属于不同层级的评测。

## 实际运行

运行 K 参数评测前：

1. 在项目根目录激活 `.venv`；
2. 确保本机 Ollama 正在运行，并且已存在 `nomic-embed-text`；
3. 不需要启动 `main.py`，因为 `--retrieval-only` 不会调用 FastAPI、Agent、联网工具或远程 LLM。

先用 3 题确认环境正常：

```powershell
& .\.venv\Scripts\python.exe .\evaluation\evaluate.py `
  --retrieval-only `
  --questions .\evaluation\retrieval_k_tuning_50.json `
  --split development `
  --limit 3 `
  --k-values 20 `
  --top-k 5
```

再用开发集比较多个 K：

```powershell
& .\.venv\Scripts\python.exe .\evaluation\evaluate.py `
  --retrieval-only `
  --questions .\evaluation\retrieval_k_tuning_50.json `
  --split development `
  --k-values 5 10 15 20 `
  --top-k 5
```

脚本会先预热 Ollama、FAISS 和 Reranker；预热不计入成绩和正式耗时。结果会自动保存到 `evaluation/results/`，包含每题的来源命中、证据命中、候选数、回退情况以及平均/中位数/P95 耗时。

## 旧题集

`questions.json` 保留为历史测试记录。它混有本地检索题和联网搜索题，并包含当前未入库的电子商务法目标来源，所以不适合作为本轮 `k` 参数选择的唯一依据。
