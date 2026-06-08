"""
文档加载模块：支持 txt / pdf / docx 多格式文档读取
"""
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from log_config import logger

def load_documents(file_paths: list[str]):
    """
    批量加载文档，增加异常捕获与日志
    :param file_paths: 文件路径列表
    :return: 解析后的Document列表
    """
    all_docs = []
    for path in file_paths:
        try:
            if not os.path.exists(path):
                logger.warning(f"文件不存在，跳过: {path}")
                continue

            ext = os.path.splitext(path)[1].lower()
            loader = None

            if ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
            else:
                logger.warning(f"不支持文件格式 {ext}，跳过 {path}")
                continue

            docs = loader.load()
            all_docs.extend(docs)
            logger.info(f"成功加载文件: {path}，段落数: {len(docs)}")

        except Exception as e:
            logger.error(f"文件 {path} 加载失败，异常: {str(e)}")
            continue
    return all_docs