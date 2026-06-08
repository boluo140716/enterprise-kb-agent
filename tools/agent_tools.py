"""
Agent 自定义工具集合：知识库检索、联网搜索、文档保存
"""
from langchain.tools import tool
from tavily import TavilyClient
from settings import TAVILY_API_KEY, SAVE_SUMMARY_PATH
from retriever import multi_hybrid_retrieve
from utils import format_retrieve_docs

# 初始化联网搜索客户端
tavily = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def search_knowledge_base(query: str) -> str:
    """
    企业本地知识库检索工具，读取PDF/DOCX/TXT内部文档
    :param query: 用户检索关键词/问题
    """
    docs = multi_hybrid_retrieve(query)
    if not docs:
        return ""
    return format_retrieve_docs(docs)

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
    将总结内容导出到本地txt文件
    :param summary_text: 待保存的总结文本
    """
    with open(SAVE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return f"执行完成：总结已保存至 {SAVE_SUMMARY_PATH}"

# 对外导出工具列表
tool_list = [search_knowledge_base, search_online, save_summary_to_txt]