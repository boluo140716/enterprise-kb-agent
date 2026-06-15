"""
会话工具函数：UUID 生成、摘要目录管理、答案提取
"""
import os
import uuid
import shutil
from langchain_core.messages import AIMessage
from settings import TEMP_SUMMARY_DIR
from log_config import logger


def _ensure_session_id(session_state: dict) -> str:
    """确保会话有唯一 ID，用于摘要目录隔离"""
    sid = session_state.get("session_id")
    if not sid:
        sid = uuid.uuid4().hex[:12]
        session_state["session_id"] = sid
    return sid


def _get_summary_dir(session_id: str) -> str:
    """获取当前会话的摘要目录路径"""
    return os.path.join(TEMP_SUMMARY_DIR, session_id)


def _cleanup_summary_dir(session_id: str):
    """删除会话的临时摘要目录"""
    dirpath = _get_summary_dir(session_id)
    if os.path.isdir(dirpath):
        try:
            shutil.rmtree(dirpath)
            logger.info(f"已删除会话摘要目录: {dirpath}")
        except Exception as e:
            logger.warning(f"删除摘要目录失败: {dirpath} — {e}")


def _extract_answer(result):
    """从 graph 最终状态提取可读答案，防御 tool_call 残留"""
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls and not last_msg.content:
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return msg.content
        return "抱歉，未能生成有效回答，请重试。"
    return last_msg.content or "（空回答）"
