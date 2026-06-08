"""
通用工具函数模块
"""

def format_retrieve_docs(docs) -> str:
    """
    将检索到的文档列表拼接为纯文本字符串
    """
    return "\n\n".join(doc.page_content for doc in docs)