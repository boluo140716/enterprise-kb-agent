"""
文件上传/清空处理：校验、加载、分片、索引 Chroma
"""
import os
from langchain_chroma import Chroma
from settings import UPLOAD_MAX_FILE_SIZE_MB, UPLOAD_MAX_FILE_COUNT
from document.loader import load_documents
from document.splitter import detail_splitter
from document.vector_store import embeddings
from log_config import logger
from web.session_utils import _cleanup_summary_dir


def handle_upload(files, session_state):
    """
    处理用户上传文件：校验 → 加载 → 分片 → 创建内存 Chroma
    文件不落盘，仅存于当前会话内存 Chroma
    """
    if session_state is None:
        session_state = {"chroma": None, "file_names": [], "file_summaries": []}

    if files is None:
        return session_state, "⚠️ 未选择文件，请先选择文档后上传"

    # 统一为列表
    if not isinstance(files, list):
        files = [files]

    # 数量限制
    if len(files) > UPLOAD_MAX_FILE_COUNT:
        return session_state, f"❌ 单次最多上传 {UPLOAD_MAX_FILE_COUNT} 个文件，当前选择了 {len(files)} 个"

    existing_names = set(session_state.get("file_names", []))
    new_files = []       # (original_name, temp_path) 有效的新文件
    skip_names = []      # 重名跳过的文件名
    over_size_names = [] # 超大的文件名
    status_lines = []

    # 第一轮：逐个校验
    for f in files:
        try:
            # Gradio 6.x gr.File 返回格式兼容：dict 或 str
            if isinstance(f, dict):
                file_name = f.get("name", os.path.basename(f.get("path", "")))
                file_path = f.get("path", "")
            elif isinstance(f, str):
                file_name = os.path.basename(f)
                file_path = f
            else:
                logger.warning(f"未知文件格式: {type(f)}")
                continue

            if not file_path or not os.path.exists(file_path):
                logger.warning(f"文件路径无效: {file_path}")
                continue

            # 大小校验
            file_size_bytes = os.path.getsize(file_path)
            if file_size_bytes > UPLOAD_MAX_FILE_SIZE_MB * 1024 * 1024:
                over_size_names.append(file_name)
                continue

            # 重名校验
            if file_name in existing_names:
                skip_names.append(file_name)
                continue

            new_files.append((file_name, file_path))
            existing_names.add(file_name)

        except Exception as e:
            logger.error(f"文件校验异常: {e}", exc_info=True)
            continue

    # 校验反馈
    if over_size_names:
        status_lines.append(
            f"❌ 以下文件超过 {UPLOAD_MAX_FILE_SIZE_MB}MB 限制，已跳过：{', '.join(over_size_names)}"
        )
    if skip_names:
        status_lines.append(f"⚠️ 以下文件已存在，已跳过：{', '.join(skip_names)}")

    if not new_files:
        if not status_lines:
            status_lines.append("⚠️ 没有有效的文件需要上传")
        return session_state, "\n".join(status_lines)

    # 第二轮：加载文档并分片
    all_chunks = []
    success_files = []
    fail_files = []
    file_summaries = list(session_state.get("file_summaries", []))

    for file_name, file_path in new_files:
        try:
            docs = load_documents([file_path])
            if not docs:
                fail_files.append(file_name)
                continue

            # 生成摘要（拼接所有页面后取前 120 字符）
            full_text = "".join(doc.page_content for doc in docs)
            summary = full_text[:120].replace("\n", " ").strip()
            if len(full_text) > 120:
                summary += "..."

            # 精细分片（记录单文件增量分片数）
            before_count = len(all_chunks)
            for doc in docs:
                chunks = detail_splitter.split_text(doc.page_content)
                all_chunks.extend(chunks)
            file_chunk_count = len(all_chunks) - before_count

            success_files.append(file_name)
            file_summaries.append({"name": file_name, "summary": summary})
            logger.info(f"上传文件解析成功: {file_name}, 分片数: {file_chunk_count}")

        except Exception as e:
            logger.error(f"文件 {file_name} 解析失败: {e}", exc_info=True)
            fail_files.append(file_name)

    if fail_files:
        status_lines.append(f"❌ 以下文件解析失败：{', '.join(fail_files)}")

    if not all_chunks:
        status_lines.append("❌ 所有文件解析后均无有效内容")
        return session_state, "\n".join(status_lines)

    # 第三轮：创建/更新内存 Chroma（增量追加，避免 O(n²) 重建）
    try:
        existing_chroma = session_state.get("chroma")
        if existing_chroma is not None:
            # 已有 Chroma：增量添加新分片，不重建
            existing_chroma.add_texts(texts=all_chunks)
            new_chroma = existing_chroma
        else:
            # 首次上传：创建新 Chroma
            new_chroma = Chroma.from_texts(
                texts=all_chunks,
                embedding=embeddings,
            )
    except Exception as e:
        logger.error(f"创建/更新内存 Chroma 失败: {e}", exc_info=True)
        return session_state, f"❌ 创建临时向量库失败：{str(e)}"

    # 更新会话状态
    session_state["chroma"] = new_chroma
    session_state["file_names"] = list(existing_names)
    session_state["file_summaries"] = file_summaries

    # 构建反馈信息
    status_lines.append(f"✅ 已索引 {len(success_files)} 个文件，共 {len(all_chunks)} 个分片")
    for fi in file_summaries:
        if fi["name"] in [s[0] for s in new_files]:
            status_lines.append(f"📄 **{fi['name']}** — {fi['summary']}")

    return session_state, "\n".join(status_lines)


def clear_upload(session_state):
    """仅清空上传的临时文件，保留对话记录"""
    try:
        if session_state:
            session_state["chroma"] = None
            session_state["file_names"] = []
            session_state["file_summaries"] = []
        logger.info("临时文档已清空")
    except Exception as e:
        logger.error(f"清空临时文档异常: {e}", exc_info=True)
    return session_state, "📭 已清空全部临时文档"


def clear_all(session_state):
    """清空对话记录 + 临时上传文件 + 输入框，删除临时摘要目录"""
    try:
        # 先清理磁盘上的临时摘要目录
        sid = session_state.get("session_id", "") if session_state else ""
        if sid:
            _cleanup_summary_dir(sid)

        if session_state:
            session_state.clear()
        logger.info("已清空全部对话与临时文档")
    except Exception as e:
        logger.error(f"清空全部状态异常: {e}", exc_info=True)
    # 返回空聊天记录、重置的 session_state、上传状态、输入框
    new_state = {"chroma": None, "file_names": [], "file_summaries": []}
    return [], new_state, "📭 已清空对话与临时文档", ""
