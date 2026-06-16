"""
检索聚合层：双层分级RAG + 向量/关键词混合检索 + LRU 内存缓存
上层 Agent 只调用此模块，不感知底层细节
"""
import copy
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from settings import TOP_K_SUB_RETRIEVE, ENSEMBLE_WEIGHT_VECTOR, ENSEMBLE_WEIGHT_BM25
from document.vector_store import faiss_search, index2full, embeddings
from document.splitter import detail_splitter
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

    # 对命中文档做精细分片
    all_chunks = []
    for idx in valid_index:
        if idx in index2full:
            full_text = index2full[idx]
        else:
            full_text = ""
        # 空文本直接跳过分片
        if not full_text.strip():
            continue
        chunks = detail_splitter.split_text(full_text)
        all_chunks.extend(chunks)

    if not all_chunks:
        return []

    # 构建混合检索器
    chroma = Chroma.from_texts(all_chunks, embedding=embeddings)
    vec_ret = chroma.as_retriever(search_kwargs={"k": TOP_K_SUB_RETRIEVE})
    bm25_ret = BM25Retriever.from_texts(all_chunks)
    bm25_ret.k = TOP_K_SUB_RETRIEVE

    hybrid_ret = EnsembleRetriever(
        retrievers=[vec_ret, bm25_ret],
        weights=[ENSEMBLE_WEIGHT_VECTOR, ENSEMBLE_WEIGHT_BM25]
    )
    return copy.deepcopy(hybrid_ret.invoke(query))
