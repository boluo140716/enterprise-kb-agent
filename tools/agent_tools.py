"""
Agent 自定义工具集合：知识库检索、联网搜索、文档保存
"""
import os
from langchain.tools import tool
from tavily import TavilyClient
from duckduckgo_search import DDGS
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
    全网实时资讯检索，用于本地无数据的实时政策、日期、赛事。
    Tavily 优先（结构化结果），DuckDuckGo 兜底（免费、覆盖面广）。
    :param query: 联网搜索关键词
    """
    res_text = ""

    # 1. Tavily 优先（结构化结果 + AI 答案）
    try:
        resp = tavily.search(query=query, search_depth="advanced", include_answer=True, max_results=5)
        answer = resp.get("answer", "")
        if answer:
            res_text += f"【Tavily 直接答案】{answer}\n\n"
        for item in resp.get("results", []):
            title = item.get("title", "无标题")
            content = item.get("content", "")
            res_text += f"【标题】{title}\n【内容】{content}\n\n"
        if res_text.strip():
            logger.info(f"Tavily 搜索成功: {query[:50]}... ({len(resp.get('results', []))} 条)")
    except Exception as e:
        logger.warning(f"Tavily 搜索失败: {type(e).__name__}: {e}")

    # 2. DuckDuckGo 兜底（Tavily 无结果或失败时补充）
    if not res_text.strip():
        try:
            ddg_results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    ddg_results.append(r)
            if ddg_results:
                res_text += "【DuckDuckGo 搜索结果】\n\n"
                for r in ddg_results:
                    res_text += f"【标题】{r.get('title', '无标题')}\n【链接】{r.get('href', '')}\n【内容】{r.get('body', '')}\n\n"
                logger.info(f"DuckDuckGo 搜索成功: {query[:50]}... ({len(ddg_results)} 条)")
        except ImportError:
            logger.warning("duckduckgo_search 未安装，请运行: pip install duckduckgo-search")
        except Exception as e:
            logger.warning(f"DuckDuckGo 搜索失败: {type(e).__name__}: {e}")

    if not res_text.strip():
        return "[联网搜索暂时不可用] 所有搜索源均失败，请基于本地知识库回答，或稍后重试。"

    return res_text


@tool
def save_summary_to_txt(summary_text: str) -> str:
    """
    将总结内容暂存，前端会弹出下载对话框让用户选择保存路径。

    白名单门禁：仅当 Web 层检测到用户输入包含保存关键词时才允许执行，
    否则直接返回拒绝提示，LLM 收到后应转为纯文本回答。
    :param summary_text: 待保存的总结文本
    """
    # 白名单门禁：Web 层未放行 → 拒绝执行
    if not session_store.get_save_allowed():
        logger.info("save_summary_to_txt 被门禁拦截（非保存类提问），拒绝执行")
        return "[拒绝] 当前提问不需要保存文档。请直接在回答中输出文本结果，不要再调用保存工具。"

    session_id = session_store.get_current_session_id()
    if session_id:
        # 写入文件（Gradio 和 FastAPI 不同进程，文件是唯一共享存储）
        summary_dir = session_store.get_summary_dir() or os.path.join(TEMP_SUMMARY_DIR, session_id)
        os.makedirs(summary_dir, exist_ok=True)
        filepath = os.path.join(summary_dir, "summary.txt")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(summary_text)
            logger.info(f"摘要已保存至文件: {filepath}")
            return "[保存成功] 文件已生成，请在界面中点击下载按钮保存到本地。"
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return f"[保存失败] {e}"
    else:
        return "[保存失败] 无法获取当前会话，请重试。"


# 对外导出工具列表
tool_list = [search_knowledge_base, search_online, save_summary_to_txt]