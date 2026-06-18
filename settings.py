"""
全局运行配置：路径、模型参数、检索参数、分片参数
"""
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# ===================== 文件路径配置 =====================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(_BASE_DIR, "first_faiss.index")
MAPPING_JSON_PATH = os.path.join(_BASE_DIR, "index_mapping.json")
TEMP_SUMMARY_DIR = "temp_summary"      # 摘要按会话ID存放，不参与知识库检索

# ===================== LLM & Embedding 模型配置 =====================
# LLM: DeepSeek（云端大模型）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL_NAME = "deepseek-chat"
LLM_TEMPERATURE = 0.3

# Embedding: 阿里百炼（DashScope）
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY", "")
EMBED_MODEL_NAME = "text-embedding-v3"

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
TOP_K_RERANK = 3                # Rerank 后保留文档数

# ===================== 上传文件参数 =====================
UPLOAD_MAX_FILE_SIZE_MB = 10        # 单文件上限 10MB
UPLOAD_MAX_FILE_COUNT = 5           # 单次最多 5 个文件
UPLOAD_TOP_K_TEMP = 3               # 临时文档 Chroma 检索返回数

# ===================== 智能体循环控制 =====================
MAX_TOOL_ROUNDS = 5                  # ReAct 最大工具调用轮数，预留多步搜索空间

# ===================== 第三方服务密钥 =====================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    import logging
    logging.getLogger("KB-Agent").warning(
        "⚠️  TAVILY_API_KEY 未设置！联网搜索功能将不可用。"
        " 请在 .env 文件中设置 TAVILY_API_KEY=<your-key>。"
    )

