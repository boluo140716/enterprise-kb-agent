"""
Pydantic 请求/响应模型，独立于业务逻辑
"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """流式对话请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户提问内容")
    session_id: Optional[str] = Field(None, description="会话 ID，首次为空则自动创建")


class ChatResponse(BaseModel):
    """非流式回退响应"""
    answer: str
    session_id: str
    download_url: Optional[str] = Field(None, description="保存操作后返回的文件下载链接")


class UploadResponse(BaseModel):
    """上传结果"""
    status: str
    message: str
    file_count: int
    chunk_count: int


class SessionClearResponse(BaseModel):
    """会话清空结果"""
    message: str


class HealthResponse(BaseModel):
    """健康检查"""
    status: str
    service: str