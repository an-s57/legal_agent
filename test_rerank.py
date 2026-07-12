"""临时脚本：对比 FAISS vs Rerank 的检索质量"""
from rag.retriever import _get_faiss_db, _rerank, _get_reranker

_get_reranker()  # 先加载模型

faiss_db = _get_faiss_db()
query = "买到假货怎么索赔"

# 没有 rerank
docs_all = faiss_db.similarity_search(query, k=5)
print("=== FAISS 直接 Top 5（无 rerank）===")
for i, d in enumerate(docs_all):
    print(f'{i+1}. [{d.metadata["source"]}] {d.page_content[:80]}...')

# 有 rerank
docs_20 = faiss_db.similarity_search(query, k=20)
docs_reranked = _rerank(query, docs_20, top_k=5)
print()
print("=== Rerank 后 Top 5 ===")
for i, d in enumerate(docs_reranked):
    print(f'{i+1}. [{d.metadata["source"]}] {d.page_content[:80]}...')