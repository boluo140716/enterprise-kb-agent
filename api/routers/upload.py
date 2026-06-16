"""
文件上传路由：临时文档管理
"""
import os
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from langchain_chroma import Chroma

from api.models import UploadResponse, SessionClearResponse
from api.dependency import inject_session, inject_chroma, cleanup_session
from document.loader import load_documents
from document.splitter import detail_splitter
from document.vector_store import embeddings
from web.session_utils import _cleanup_summary_dir
from settings import UPLOAD_MAX_FILE_SIZE_MB, UPLOAD_MAX_FILE_COUNT
from log_config import logger

router = APIRouter(prefix="/api/upload", tags=["文件上传"])

# 内存会话存储（生产环境应替换为 Redis）
# key: session_id, value: {"chroma": Chroma, "file_names": list[str]}
_upload_sessions: dict = {}


def _get_session_store(session_id: str) -> dict:
    """获取或创建会话存储"""
    if session_id not in _upload_sessions:
        _upload_sessions[session_id] = {"chroma": None, "file_names": []}
    return _upload_sessions[session_id]


@router.post("", response_model=UploadResponse)
async def upload_files(
    session_id: str = Form(default="", description="会话 ID"),
    files: list[UploadFile] = File(..., description="文档文件"),
):
    """
    上传文档到临时向量库。

    支持格式：pdf, docx, txt, md, xlsx, xls
    """
    session_id, _ = inject_session(session_id)
    store = _get_session_store(session_id)

    # 数量校验
    if len(files) > UPLOAD_MAX_FILE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多上传 {UPLOAD_MAX_FILE_COUNT} 个文件"
        )

    # 大小校验（UploadFile 没有 size 属性，用文件内容长度近似）
    valid_files = []
    for f in files:
        content = await f.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > UPLOAD_MAX_FILE_SIZE_MB:
            logger.warning(f"文件 {f.filename} 超过 {UPLOAD_MAX_FILE_SIZE_MB}MB 限制，跳过")
            continue
        valid_files.append((f.filename, content))
        await f.seek(0)  # 重置指针

    if not valid_files:
        raise HTTPException(status_code=400, detail="没有有效的文件可供上传")

    # 写入临时文件（load_documents 需要文件路径）
    temp_dir = f"temp_upload_{session_id}"
    os.makedirs(temp_dir, exist_ok=True)
    temp_paths = []

    try:
        for filename, content in valid_files:
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "wb") as f:
                f.write(content)
            temp_paths.append(temp_path)

        # 加载 + 分片
        docs = load_documents(temp_paths)
        all_chunks = []
        for doc in docs:
            chunks = detail_splitter.split_text(doc.page_content)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="文档解析后无有效内容")

        # 增量追加到 Chroma
        existing_chroma = store.get("chroma")
        if existing_chroma is not None:
            existing_chroma.add_texts(texts=all_chunks)
        else:
            existing_chroma = Chroma.from_texts(texts=all_chunks, embedding=embeddings)
            store["chroma"] = existing_chroma

        file_names = [fn for fn, _ in valid_files if fn not in store["file_names"]]
        store["file_names"].extend(file_names)

        # 注入 ContextVar
        inject_chroma(store["chroma"], store["file_names"])

        return UploadResponse(
            status="success",
            message=f"已索引 {len(valid_files)} 个文件，共 {len(all_chunks)} 个分片",
            file_count=len(valid_files),
            chunk_count=len(all_chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传失败: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
    finally:
        # 清理临时文件
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass
        cleanup_session()


@router.delete("", response_model=SessionClearResponse)
async def clear_upload(session_id: str = Form(default="", description="会话 ID")):
    """清空当前会话的上传文件"""
    if session_id and session_id in _upload_sessions:
        del _upload_sessions[session_id]
        _cleanup_summary_dir(session_id)
        return SessionClearResponse(message="已清空全部上传文件与摘要")
    return SessionClearResponse(message="会话不存在或已清空")