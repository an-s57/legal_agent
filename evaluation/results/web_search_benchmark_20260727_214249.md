# 联网法律搜索服务对比报告

- 运行时间：2026-07-27T21:42:49
- 题集：`D:\legal_agent\evaluation\web_search_benchmark_20.json`（20 题）
- 统一查询后缀：`法律法规 中国`
- 每题返回数：Top-3
- 边界：本报告只比较网页搜索服务；不经过 Agent、LLM、FAISS 或 Reranker。

## 自动指标汇总

| 服务 | 成功率 | 官方来源命中@N | 核心主题命中@N | 自动通过@N | 细节摘要覆盖（平均） | 平均 / 中位数 / P95 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DuckDuckGo（ddgs） | 20/20 (100.0%) | 9/20 (45.0%) | 16/20 (80.0%) | 8/20 (40.0%) | 82.5% | 4241 / 4059 / 5428 ms |
| AnySearch | 20/20 (100.0%) | 19/20 (95.0%) | 20/20 (100.0%) | 19/20 (95.0%) | 57.5% | 1517 / 1491 / 2210 ms |

## 指标如何理解

- **官方来源命中@N**：Top-N 中至少有一个链接属于该题预标注的官方域名。
- **核心主题命中@N**：Top-N 的标题或摘要覆盖该题的核心法律名称、办事渠道或纠纷主题。
- **细节摘要覆盖**：条文编号、金额、地域、具体事实等是否出现在搜索摘要中；只作诊断，不作为通过门槛。
- **自动通过@N**：同时满足官方来源命中与核心主题命中；它是可复核的代理指标，不等同于法律意见正确。

## 建议人工复核（仅看有分歧或双方失败的题）

| 题号 | 原因 | DuckDuckGo（ddgs） 首条链接 | AnySearch 首条链接 |
| --- | --- | --- | --- |
| colloquial-02 | 不同服务的自动通过结果不一致 | https://m.haolvshi.com.cn/ztw/0-72399.html | https://rsj.beijing.gov.cn/xxgk/gzdt/202108/t20210819_2472135.html |
| colloquial-03 | 不同服务的自动通过结果不一致 | https://www.dutenews.com/n/article/6830955 | https://www.ahjd.gov.cn/OpennessContent/show/2650251.html |
| colloquial-04 | 不同服务的自动通过结果不一致 | https://www.dutenews.com/n/article/10563435 | https://www.gov.cn/zhengce/2020-11/03/content_5723721.htm |
| policy-01 | 不同服务的自动通过结果不一致 | https://www.moj.gov.cn/pub/sfbgw/flfggz/flfggzbmgz/201701/t20170124_145952.html | http://xzfg.moj.gov.cn/front/law/detail?LawID=1708 |
| policy-03 | 不同服务的自动通过结果不一致 | https://amsdottorato.unibo.it/id/eprint/11763/1/LUAN_ZHIBO_TESI.pdf | https://www.court.gov.cn/zixun/xiangqing/6042.html |
| policy-04 | 不同服务的自动通过结果不一致 | https://m.nj.bendibao.com/live/178187.shtm | https://rsj.beijing.gov.cn/xxgk/2024zcwj/202507/t20250725_4158456.html |

## 结论填写模板

先根据自动通过率、失败率和 P95 耗时选出候选服务；再查看上表中少量分歧题。`duckduckgo-search` 仅作为当前生产接入基线；若新版 ddgs 仍无明显改善，而 AnySearch 显著更稳定，再考虑替换生产联网工具。
