"""
检索聚合层：双层分级RAG + 向量/关键词混合检索 + LRU 内存缓存
上层 Agent 只调用此模块，不感知底层细节
"""
import copy
from functools import lru_cache
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from settings import TOP_K_SUB_RETRIEVE, ENSEMBLE_WEIGHT_VECTOR, ENSEMBLE_WEIGHT_BM25, TOP_K_RERANK
from document.vector_store import faiss_search, index2full, embeddings
from document.splitter import detail_splitter
from document.reranker import rerank_documents
from log_config import logger


@lru_cache(maxsize=128)
def multi_hybrid_retrieve(query: str):
    """
    带 LRU 缓存的双层混合检索。
    相同查询直接返回缓存结果，跳过 FAISS / Chroma 向量计算。

    检索流程：
    1. FAISS 摘要粗筛 → 锁定相关文档
    2. 文档精细分片（500 字符）
    3. Chroma 向量 + BM25 关键词 Ensemble 混合检索
    """
    logger.info(f"检索(缓存未命中): {query[:80]}...")

    valid_index = faiss_search(query)
    if not valid_index:
        return []

    # 对命中文档做精细分片（保留分片索引，用于后续扩展上下文）
    all_chunks = []
    chunk_index_map = []  # 记录每个分片所属文档的索引
    for idx in valid_index:
        if idx in index2full:
            full_text = index2full[idx]
        else:
            full_text = ""
        if not full_text.strip():
            continue
        chunks = detail_splitter.split_text(full_text)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_index_map.append(idx)  # 标记分片来源文档

    if not all_chunks:
        return []

    # 建立分片相邻关系：同一文档内相邻分片互为上下文
    chunk_neighbors = {}
    for i, idx in enumerate(chunk_index_map):
        neighbors = []
        if i > 0 and chunk_index_map[i - 1] == idx:
            neighbors.append(all_chunks[i - 1])
        if i + 1 < len(chunk_index_map) and chunk_index_map[i + 1] == idx:
            neighbors.append(all_chunks[i + 1])
        chunk_neighbors[i] = neighbors

    # 混合检索：向量 + BM25，各取 TOP_K_SUB_RETRIEVE 条，Ensemble 合并
    # 为 Rerank 提供更多候选，ensemble k 设为 TOP_K_SUB_RETRIEVE * 2
    chroma = Chroma.from_texts(all_chunks, embedding=embeddings)
    vec_ret = chroma.as_retriever(search_kwargs={"k": TOP_K_SUB_RETRIEVE})
    bm25_ret = BM25Retriever.from_texts(all_chunks)
    bm25_ret.k = TOP_K_SUB_RETRIEVE

    hybrid_ret = EnsembleRetriever(
        retrievers=[vec_ret, bm25_ret],
        weights=[ENSEMBLE_WEIGHT_VECTOR, ENSEMBLE_WEIGHT_BM25]
    )
    hybrid_docs = copy.deepcopy(hybrid_ret.invoke(query))

    # Rerank 精排：对混合检索结果重排序，取 TOP_K_RERANK 条最相关文档
    if hybrid_docs and len(hybrid_docs) > TOP_K_RERANK:
        hybrid_docs = rerank_documents(query, hybrid_docs)

    # 分片上下文扩展：将命中分片的前后相邻分片也加入结果，恢复文档上下文
    expanded = list(hybrid_docs)
    seen_contents = {d.page_content for d in expanded}
    for doc in hybrid_docs:
        # 找到该分片在 all_chunks 中的位置
        for i, chunk in enumerate(all_chunks):
            if chunk == doc.page_content and i in chunk_neighbors:
                for neighbor in chunk_neighbors[i]:
                    if neighbor not in seen_contents:
                        seen_contents.add(neighbor)
                        expanded.append(Document(page_content=neighbor))
                break

    return expanded