"""
Agent 自定义工具集合：知识库检索、联网搜索、文档保存
"""
import os
from langchain.tools import tool
from tavily import TavilyClient
from settings import TAVILY_API_KEY, TEMP_SUMMARY_DIR, UPLOAD_TOP_K_TEMP
from retriever import multi_hybrid_retrieve
from utils import format_retrieve_docs
from log_config import logger
import session_store

# 初始化联网搜索客户端
tavily = TavilyClient(api_key=TAVILY_API_KEY)


def _deduplicate_docs(docs: list) -> list:
    """基于 page_content 去重，保留首次出现的文档"""
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


@tool
def search_knowledge_base(query: str) -> str:
    """
    企业本地知识库检索工具，读取PDF/DOCX/TXT内部文档
    :param query: 用户检索关键词/问题
    """
    # 1. 全局 FAISS 持久知识库检索
    faiss_docs = multi_hybrid_retrieve(query)

    # 2. 当前会话临时 Chroma 检索（用户上传文档）
    temp_docs = []
    try:
        temp_chroma = session_store.get_current_chroma()
        if temp_chroma is not None:
            temp_docs = temp_chroma.similarity_search(query, k=UPLOAD_TOP_K_TEMP)
    except Exception as e:
        logger.error(f"临时文档 Chroma 检索异常（不影响全局检索）: {e}")

    # 3. 合并去重：临时文档排前（用户刚上传更相关），全局 FAISS 排后
    all_docs = temp_docs + (faiss_docs if faiss_docs else [])
    if not all_docs:
        return ""
    all_docs = _deduplicate_docs(all_docs)
    return format_retrieve_docs(all_docs)


@tool
def search_online(query: str) -> str:
    """
    全网实时资讯检索，用于本地无数据的实时政策、日期、赛事
    :param query: 联网搜索关键词
    """
    resp = tavily.search(query=query)
    res_text = ""
    for item in resp["results"]:
        res_text += f"【标题】{item['title']}\n【内容】{item['content']}\n\n"
    return res_text


@tool
def save_summary_to_txt(summary_text: str) -> str:
    """
    将总结内容导出到当前会话的临时摘要目录（temp_summary/<session_id>/summary.txt）
    同时将完整文本存入 ContextVar 供 Web 页内预览。

    白名单门禁：仅当 Web 层检测到用户输入包含保存关键词时才允许执行，
    否则直接返回拒绝提示，LLM 收到后应转为纯文本回答。
    :param summary_text: 待保存的总结文本
    """
    # 白名单门禁：Web 层未放行 → 拒绝执行
    if not session_store.get_save_allowed():
        logger.info("save_summary_to_txt 被门禁拦截（非保存类提问），拒绝执行")
        return "[拒绝] 当前提问不需要保存文档。请直接在回答中输出文本结果，不要再调用保存工具。"

    # 优先使用会话级目录（Web 多标签隔离）
    session_dir = session_store.get_summary_dir()

    if session_dir:
        os.makedirs(session_dir, exist_ok=True)
        filepath = os.path.join(session_dir, "summary.txt")
    else:
        # 回退：控制台入口无会话概念，写到 temp_summary 根目录
        os.makedirs(TEMP_SUMMARY_DIR, exist_ok=True)
        filepath = os.path.join(TEMP_SUMMARY_DIR, "summary.txt")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(summary_text)
        logger.info(f"摘要已保存: {filepath}")

        # 存入 ContextVar 供 Web 预览面板读取
        session_store.set_summary_content(summary_text)

        return f"执行完成：总结已保存至 {filepath}"
    except Exception as e:
        logger.error(f"保存摘要失败: {e}", exc_info=True)
        return f"[保存失败] {type(e).__name__}: {str(e)}"


# 对外导出工具列表
tool_list = [search_knowledge_base, search_online, save_summary_to_txt]
