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
TEMP_SUMMARY_DIR = "temp_summary"      # 摘要按会话ID存放，不参与知识库检索

# ===================== LLM & Embedding 模型配置 =====================
LLM_MODEL_NAME = "qwen2:7b"
LLM_TEMPERATURE = 0.3
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

# ===================== 上传文件参数 =====================
UPLOAD_MAX_FILE_SIZE_MB = 10        # 单文件上限 10MB
UPLOAD_MAX_FILE_COUNT = 5           # 单次最多 5 个文件
UPLOAD_TOP_K_TEMP = 3               # 临时文档 Chroma 检索返回数

# ===================== 智能体循环控制 =====================
MAX_TOOL_ROUNDS = 3                  # ReAct 最大工具调用轮数，超出强制 LLM 文本回答

# ===================== 第三方服务密钥 =====================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 内存缓存：相同文本只向量化一次，减少90%网络请求
set_llm_cache(InMemoryCache())