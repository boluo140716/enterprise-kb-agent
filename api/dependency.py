"""
会话管理工具：ContextVar 注入 + 清理
"""
import os
import uuid
from settings import TEMP_SUMMARY_DIR
import session_store
from log_config import logger


def inject_session(session_id: str | None, save_allowed: bool = False) -> tuple[str, str]:
    """
    注入当前请求的会话上下文到 ContextVar。

    :param session_id: 会话 ID（为空则自动生成）
    :param save_allowed: 是否允许保存工具
    :return: (session_id, summary_dir)
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:12]

    summary_dir = os.path.join(TEMP_SUMMARY_DIR, session_id)

    session_store.set_summary_dir(summary_dir)
    session_store.set_save_allowed(save_allowed)
    session_store.set_current_session_id(session_id)

    return session_id, summary_dir


def inject_chroma(chroma, file_names: list):
    """注入临时文档 Chroma 到 ContextVar"""
    if chroma is not None:
        session_store.set_current(chroma, file_names)


def cleanup_session():
    """请求结束后清理 ContextVar"""
    try:
        session_store.clear_current()
    except Exception as e:
        logger.warning(f"清理 ContextVar 异常: {e}")