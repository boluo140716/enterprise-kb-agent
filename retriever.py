"""
检索聚合层：双层分级RAG + 向量/关键词混合检索
上层Agent只调用此模块，不感知底层细节
"""
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from settings import TOP_K_SUB_RETRIEVE, ENSEMBLE_WEIGHT_VECTOR, ENSEMBLE_WEIGHT_BM25
from document.vector_store import faiss_search, index2full, embeddings
from document.splitter import detail_splitter
from utils import format_retrieve_docs
import time

def multi_hybrid_retrieve(query: str):
    """
    双层RAG主逻辑：
    1. FAISS摘要粗筛完整文档
    2. 文档精细分片
    3. Chroma向量 + BM25关键词 混合检索
    """
    valid_index = faiss_search(query)
    time.sleep(0.15)
    if not valid_index:
        return []

    # 对命中文档做精细分片
    all_chunks = []
    for idx in valid_index:
        full_text = index2full[idx]
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
    return hybrid_ret.invoke(query)