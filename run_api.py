"""
FastAPI 服务启动入口

启动方式（在项目根目录执行）：
    python run_api.py
"""
import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=7863,
        reload=True,
        reload_dirs=["api", "agent", "tools", "document", "web"],
        log_level="warning",
    )