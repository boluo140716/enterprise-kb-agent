"""
文档加载模块：支持 txt / pdf / docx / markdown / excel 多格式文档读取
"""
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader,UnstructuredMarkdownLoader,UnstructuredExcelLoader
from log_config import logger


def _load_txt_with_fallback(path: str):
    """尝试多种编码加载 txt 文件（UTF-8 → GBK → latin-1）"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
    for enc in encodings:
        try:
            loader = TextLoader(path, encoding=enc)
            docs = loader.load()
            if enc != "utf-8" and enc != "utf-8-sig":
                logger.info(f"txt 文件使用 {enc} 编码加载: {path}")
            return docs
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 最终回退
    logger.warning(f"所有编码尝试失败，使用 latin-1 回退: {path}")
    return TextLoader(path, encoding="latin-1").load()


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
                docs = _load_txt_with_fallback(path)
            elif ext == ".pdf":
                docs = PyPDFLoader(path).load()
            elif ext == ".docx":
                docs = Docx2txtLoader(path).load()
            elif ext == ".md":
                docs = UnstructuredMarkdownLoader(path).load()
            elif ext in (".xlsx", ".xls"):
                docs = UnstructuredExcelLoader(path, mode="elements").load()
            else:
                logger.warning(f"不支持文件格式 {ext}，跳过 {path}")
                continue

            all_docs.extend(docs)
            logger.info(f"成功加载文件: {path}，段落数: {len(docs)}")

        except Exception as e:
            logger.error(f"文件 {path} 加载失败，异常: {str(e)}")
            continue
    return all_docs