"""
会话级临时存储桥梁（ContextVar 实现，多浏览器标签页隔离）

使用 contextvars.ContextVar 而非全局单例：
- Chroma：每个标签页独立，关闭即销毁（内存模式，不落盘）
- Summary：按会话 ID 存放在 temp_summary/<session_id>/，关闭/清空自动删除
"""
import contextvars

# ---- Chroma 临时文档 ----
_current_chroma = contextvars.ContextVar('current_chroma', default=None)
_current_file_info = contextvars.ContextVar('current_file_info', default=None)

# ---- Save 白名单门禁 ----
# 由 Web 层根据用户输入关键词设置，控制 save_summary_to_txt 是否允许执行
_current_save_allowed = contextvars.ContextVar('current_save_allowed', default=False)

# 当前会话 ID（供 save_summary_to_txt 工具获取 session_id）
_current_session_id = contextvars.ContextVar('current_session_id', default=None)

# ---- Summary 摘要导出 ----
# 当前会话的摘要目录路径（temp_summary/<session_id>/）
_current_summary_dir = contextvars.ContextVar('current_summary_dir', default=None)
# 最近一次 save_summary_to_txt 生成的摘要完整文本（供 Web 预览）
_current_summary_content = contextvars.ContextVar('current_summary_content', default=None)


# ===================== Chroma =====================

def set_current(chroma, file_info):
    """注入当前会话的 Chroma 和文件信息到执行上下文"""
    _current_chroma.set(chroma)
    _current_file_info.set(file_info)


def get_current_chroma():
    """获取当前会话的 Chroma 实例，无临时文档时返回 None"""
    return _current_chroma.get()


def get_current_file_info():
    """获取当前会话已上传文件信息列表"""
    return _current_file_info.get()


# ===================== Save 白名单 =====================

def set_save_allowed(allowed: bool):
    """Web 层设置：当前提问是否允许调用保存工具"""
    _current_save_allowed.set(allowed)


def get_save_allowed() -> bool:
    """保存工具调用前检查：当前提问是否允许保存"""
    return _current_save_allowed.get()


def set_current_session_id(session_id: str):
    """设置当前会话 ID"""
    _current_session_id.set(session_id)


def get_current_session_id() -> str | None:
    """获取当前会话 ID"""
    return _current_session_id.get()


# ===================== Summary =====================

# 按会话 ID 存储摘要内容（供下载接口读取）
_summary_store: dict[str, str] = {}


def store_summary(session_id: str, content: str):
    """按会话 ID 存储摘要内容"""
    _summary_store[session_id] = content


def get_stored_summary(session_id: str) -> str | None:
    """按会话 ID 读取摘要内容"""
    return _summary_store.get(session_id)


def remove_stored_summary(session_id: str):
    """下载后清理"""
    _summary_store.pop(session_id, None)
    logger.info(f"已清理会话 {session_id} 内存摘要缓存")

def set_summary_dir(summary_dir):
    """注入当前会话的摘要存放目录"""
    _current_summary_dir.set(summary_dir)


def get_summary_dir():
    """获取当前会话的摘要存放目录"""
    return _current_summary_dir.get()


def set_summary_content(content):
    """存入最近一次摘要文本（供 Web 预览面板读取）"""
    _current_summary_content.set(content)


def get_summary_content():
    """获取最近一次摘要文本"""
    return _current_summary_content.get()


# ===================== 清理 =====================

def clear_current():
    """清除当前会话所有上下文（每次 invoke 结束后调用）"""
    _current_chroma.set(None)
    _current_file_info.set(None)
    _current_save_allowed.set(False)
    _current_summary_dir.set(None)
    _current_summary_content.set(None)

    # 同时清除 agent/nodes.py 中的 KB 文档缓存，防止跨请求数据泄露
    try:
        from agent.nodes import _kb_docs_cache
        _kb_docs_cache.set("")
    except ImportError:
        pass