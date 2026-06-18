"""
Reranker 重排序模块：对混合检索结果进行精排，提升 top-k 文档相关性

使用阿里百炼 DashScope gte-rerank 模型，与现有 Embedding 共用同一 API Key。
"""
import os
import time
from typing import List
from langchain_core.documents import Document
from log_config import logger
from settings import BAILIAN_API_KEY, TOP_K_RERANK


def rerank_documents(query: str, docs: list[Document], top_k: int = TOP_K_RERANK) -> list[Document]:
    """
    对文档列表进行重排序，返回相关性最高的 top_k 个文档。

    Args:
        query: 用户原始查询
        docs: 待重排的文档列表
        top_k: 保留文档数，默认 3

    Returns:
        相关性从高到低排列的 top_k 文档

    异常处理：
    - API 不可用时，降级返回原始文档的前 top_k 个（不阻断主流程）
    - 文档数为 0 或 ≤ top_k 时，直接返回
    """
    if not docs:
        return []

    if len(docs) <= top_k:
        return docs

    documents = [d.page_content for d in docs]

    try:
        import dashscope
        from dashscope import TextReRank

        dashscope.api_key = BAILIAN_API_KEY

        # 重试 + 退避：百炼 API 偶发限流
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                resp = TextReRank.call(
                    model="gte-rerank",
                    query=query,
                    documents=documents,
                    top_n=top_k,
                    return_documents=False,
                )
                if resp.status_code == 200 and resp.output and resp.output.get("results"):
                    # 按相关性分数降序排列
                    results = sorted(resp.output["results"], key=lambda x: x["relevance_score"], reverse=True)
                    reranked = [docs[r["index"]] for r in results]
                    logger.info(
                        f"Rerank 完成: {len(docs)} → {len(reranked)} 条, "
                        f"top 分数: {[round(r['relevance_score'], 4) for r in results]}"
                    )
                    return reranked
                else:
                    logger.warning(f"Rerank API 返回异常: {resp.status_code} {resp.message}")
                    break
            except Exception as e:
                if attempt < max_retries:
                    wait = (attempt + 1) * 1.5
                    logger.warning(f"Rerank 重试 {attempt + 1}/{max_retries}，等待 {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

    except ImportError:
        logger.error("dashscope 未安装，Rerank 不可用。请运行: pip install dashscope")
    except Exception as e:
        logger.error(f"Rerank 失败，降级返回原始结果: {e}")

    # 降级：返回原始文档前 top_k 个
    return docs[:top_k]