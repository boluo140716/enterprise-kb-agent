"""
全局日志配置模块
统一日志格式、输出级别，替代原生print
"""
import logging
import logging.handlers
import os

# 日志根目录配置（相对于本项目根目录）
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 基础日志格式
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 配置根日志（带轮转：单文件最大 10MB，保留 5 个备份）
log_path = os.path.join(LOG_DIR, "run.log")
rotating_handler = logging.handlers.RotatingFileHandler(
    log_path,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8",
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        rotating_handler,
        logging.StreamHandler()  # 控制台同时输出
    ]
)

# 对外暴露日志对象
logger = logging.getLogger("KB-Agent")