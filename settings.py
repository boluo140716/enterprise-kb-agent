"""
全局运行配置：路径、模型参数、检索参数、分片参数
"""
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
import os

# 加载环境变量
load_dotenv()

# ===================== 文件路径配置 =====================
FAISS_INDEX_PATH = "first_faiss.index"
MAPPING_JSON_PATH = "index_mapping.json"
SAVE_SUMMARY_PATH = "summary.txt"

# ===================== LLM & Embedding 模型配置 =====================
LLM_MODEL_NAME = "qwen2:7b"
LLM_TEMPERATURE = 0.8
LLM_GPU_NUM = 0
EMBED_MODEL_NAME = "all-minilm"

# ===================== 文本分片参数 =====================
ABSTRACT_CHUNK_SIZE = 80
ABSTRACT_CHUNK_OVERLAP = 0
DETAIL_CHUNK_SIZE = 500
DETAIL_CHUNK_OVERLAP = 100

# ===================== 检索参数 =====================
TOP_K_FIRST_FAISS = 3
TOP_K_SUB_RETRIEVE = 3
ENSEMBLE_WEIGHT_VECTOR = 0.6
ENSEMBLE_WEIGHT_BM25 = 0.4

# ===================== 第三方服务密钥 =====================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 内存缓存：相同文本只向量化一次，减少90%网络请求
set_llm_cache(InMemoryCache())