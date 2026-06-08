"""
全局日志配置模块
统一日志格式、输出级别，替代原生print
"""
import logging
import os

# 日志根目录配置
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 基础日志格式
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 配置根日志
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/run.log", encoding="utf-8"),
        logging.StreamHandler()  # 控制台同时输出
    ]
)

# 对外暴露日志对象
logger = logging.getLogger("KB-Agent")