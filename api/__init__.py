"""
FastAPI 应用实例 + CORS 中间件
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="企业知识库 RAG 智能问答系统",
    description="支持流式 SSE 对话、文档上传、多会话隔离的 RESTful API",
    version="1.0.0",
)

# CORS：允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "service": "enterprise-kb-agent"}

# 延迟导入路由，避免循环依赖
from api.routers import chat, upload

app.include_router(chat.router)
app.include_router(upload.router)