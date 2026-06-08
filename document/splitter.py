"""
文本分片模块：摘要分片、正文精细分片
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from settings import (
    ABSTRACT_CHUNK_SIZE,
    ABSTRACT_CHUNK_OVERLAP,
    DETAIL_CHUNK_SIZE,
    DETAIL_CHUNK_OVERLAP
)

# 摘要分片器（用于FAISS一级检索）
abstract_splitter = RecursiveCharacterTextSplitter(
    chunk_size=ABSTRACT_CHUNK_SIZE,
    chunk_overlap=ABSTRACT_CHUNK_OVERLAP
)

# 正文精细分片器（用于二级混合检索）
detail_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DETAIL_CHUNK_SIZE,
    chunk_overlap=DETAIL_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", ",", " ", ""],
    length_function=len,
    is_separator_regex=False
)