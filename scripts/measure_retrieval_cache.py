"""检索缓存「省耗时」计时器 —— 免费，不需要 DeepSeek。

它做的事：对每个问题，先删掉旧缓存保证冷启动 → 第一次真检索(计时) → 第二次命中(计时)
→ 打印「省了多少 ms」和结果是否一致，最后汇总平均收益。

前提：Ollama 在跑、向量库已建(rag/vectorstore/db_faiss)、Redis 在跑。
（Redis 没起也不会报错，只是全走 fail-open、没有命中、省耗时≈0。）

用法（激活 .venv 后，在项目根目录）：
    python scripts/measure_retrieval_cache.py
想跑更多问题：直接往下面的 QUERIES 里加，或换成从评测集读。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.legal_tools import legal_rag_search
from cache.redis_client import (
    _get_client,
    build_retrieval_key,
    reset_stats,
    get_cache_stats,
)

# 内置一批不同领域的法律问题；可自行增删
QUERIES = [
    "试用期最长可以约定几个月",
    "违法解除劳动合同怎么赔偿",
    "消费者买到假货怎么维权",
    "租房押金不退怎么办",
    "加班工资怎么计算",
    "离婚财产如何分割",
    "交通事故赔偿标准",
    "竞业限制补偿标准",
    "民间借贷利率上限是多少",
    "工伤认定需要哪些材料",
    "网络购物七天无理由退货规定",
    "物业费不交会有什么后果",
]


def _drop_cache(query: str) -> None:
    """删掉该 query 的检索缓存 key，保证本轮从「未命中」开始计时。"""
    try:
        client = _get_client()
        if client is not None:
            client.delete(build_retrieval_key(query))
    except Exception:
        pass  # fail-open：删不掉就照常跑（可能第一次就命中，只影响本条计时）


def _timed(query: str):
    start = time.perf_counter()
    result = legal_rag_search.invoke({"query": query})
    return (time.perf_counter() - start) * 1000, result


def main() -> None:
    print("== 检索缓存计时（免费，不花钱）==")
    print("先热身一次加载 reranker 模型（这条不计时）...")
    _drop_cache("__warmup__")
    legal_rag_search.invoke({"query": "热身：合同的定义"})  # 触发模型加载，避免污染首条计时

    reset_stats()
    saved_list = []
    print(f"\n[序号] 省耗时   (未命中→命中)        一致性  问题")
    for i, q in enumerate(QUERIES, 1):
        _drop_cache(q)                    # 清掉，确保第一次是真检索
        miss_ms, r1 = _timed(q)           # 未命中 → 真检索（含 rerank）+ 写缓存
        hit_ms, r2 = _timed(q)            # 命中 → 跳过检索
        saved = miss_ms - hit_ms
        saved_list.append(saved)
        ok = "OK" if r1 == r2 else "不一致!"
        print(f"[{i:>2}] 省 ~{saved:>6.0f}ms  ({miss_ms:>6.0f} -> {hit_ms:>4.0f})   {ok:<4}  {q}")

    n = len(saved_list)
    avg = sum(saved_list) / n if n else 0.0
    stats = get_cache_stats()["retrieval"]
    print("\n== 汇总 ==")
    print(f"样本数：{n}")
    print(f"平均每次命中省：~{avg:.0f} ms")
    print(f"检索缓存计数：hit={stats['hit']} miss={stats['miss']} "
          f"hit_rate={stats['hit_rate']}")
    print("\n提示：把这里的『平均省耗时』记下来，就是你面试/汇报里『检索缓存省了多少延迟』的实测数。")


if __name__ == "__main__":
    main()
