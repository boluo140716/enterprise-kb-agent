"""
FAISS向量库管理模块：索引构建、持久化、向量检索
"""
import os
import json
import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings
from settings import (
    FAISS_INDEX_PATH,
    MAPPING_JSON_PATH,
    TEMP_SUMMARY_DIR,
    EMBED_MODEL_NAME,
    TOP_K_FIRST_FAISS
)
from log_config import logger

# 初始化向量化模型
embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

# 全局FAISS缓存变量
faiss_index = None
index2abs = {}
index2full = {}

def init_faiss_store():
    """初始化/加载FAISS索引，程序启动自动执行"""
    global faiss_index, index2abs, index2full

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(MAPPING_JSON_PATH):
        logger.info("检测到FAISS缓存，直接加载本地索引")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        # json.load 会将整数键转为字符串，需要转回 int，否则 faiss_search 返回的 int 索引无法命中
        index2abs = {int(k): v for k, v in map_data["index2abs"].items()}
        index2full = {int(k): v for k, v in map_data["index2full"].items()}
        return

    logger.info("无FAISS缓存，开始全量构建索引")
    # 遍历当前目录所有文档（排除 temp_summary 摘要临时文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    supported_ext = (".txt", ".pdf", ".docx")
    file_paths = []

    for filename in os.listdir(root_dir):
        full_path = os.path.join(root_dir, filename)
        # 跳过 temp_summary 临时摘要目录
        if os.path.isdir(full_path) and filename == TEMP_SUMMARY_DIR:
            logger.info(f"跳过临时摘要目录: {TEMP_SUMMARY_DIR}")
            continue
        if filename.lower().endswith(supported_ext):
            file_paths.append(full_path)

    # 延迟导入，避免循环依赖
    from document.loader import load_documents
    from document.splitter import abstract_splitter

    all_docs = load_documents(file_paths)
    abstract_list = []
    doc_full_map = {}

    for doc in all_docs:
        full_text = doc.page_content
        abs_chunks = abstract_splitter.split_text(full_text)
        if not abs_chunks:
            continue
        abs_text = abs_chunks[0]
        abstract_list.append(abs_text)
        doc_full_map[abs_text] = full_text

    # 向量化并构建FAISS
    embed_arr = [embeddings.embed_query(text) for text in abstract_list]
    embed_np = np.array(embed_arr, dtype=np.float32)
    dim = embed_np.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embed_np)

    # 构建映射关系
    for idx, abs_text in enumerate(abstract_list):
        index2abs[idx] = abs_text
        index2full[idx] = doc_full_map[abs_text]

    # 持久化保存
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(MAPPING_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"index2abs": index2abs, "index2full": index2full}, f, ensure_ascii=False)
    logger.info("FAISS索引与映射文件保存完成")

def faiss_search(query: str) -> list[int]:
    """FAISS一级向量检索，返回有效文档下标"""
    q_vec = np.array([embeddings.embed_query(query)], dtype=np.float32)
    _, idxs = faiss_index.search(q_vec, TOP_K_FIRST_FAISS)
    hit_list = idxs[0].tolist()
    valid_index = [i for i in hit_list if i != -1]
    return valid_index

# 模块加载时自动初始化向量库
init_faiss_store()