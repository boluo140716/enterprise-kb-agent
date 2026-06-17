"""
流式对话处理：异步生成器 chat_respond，含加载占位 + token 流式 + 摘要卡片嵌入
"""
import os
import traceback
from langchain_core.messages import HumanMessage
from agent.graph_builder import agent_app
from log_config import logger
import session_store
from web.session_utils import _ensure_session_id, _get_summary_dir, _extract_answer

# 保存/导出关键词白名单（单一数据源：来自 prompts.py）
from prompts import SAVE_KEYWORDS


def _is_save_request(user_input: str) -> bool:
    """检测用户输入是否明确要求保存/导出文档"""
    return any(kw in user_input for kw in SAVE_KEYWORDS)


async def chat_respond(user_input, chat_history, session_state):
    """
    处理用户提问：流式返回 LLM 生成的回答。

    渲染顺序：
    1. 先 yield 仅含用户消息的 chat_history → 用户看到自己提问
    2. 再 yield 含加载动画占位的 chat_history
    3. 进入 astream_events 逐 token yield → 流式输出

    保存门禁：
    - 仅当用户输入包含保存关键词时，才放行 save_summary_to_txt + 嵌入摘要卡片
    - 普通问答：门禁关闭，工具拒绝执行，卡片逻辑跳过
    """
    try:
        if session_state is None:
            session_state = {"chroma": None, "file_names": [], "file_summaries": []}

        if not user_input or not user_input.strip():
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": "⚠️ 请输入有效问题"})
            yield "", chat_history, session_state
            return

        # ---- 白名单门禁：是否允许保存 ----
        save_requested = _is_save_request(user_input)

        # ---- 确保会话有唯一 ID + 摘要目录 ----
        sid = _ensure_session_id(session_state)
        summary_dir = _get_summary_dir(sid)

        # ---- ContextVar 注入：Chroma + 摘要目录 + 保存门禁（在生成循环外） ----
        chroma = session_state.get("chroma")
        if chroma is not None:
            try:
                session_store.set_current(chroma, session_state.get("file_names", []))
            except Exception as ctx_err:
                logger.error(f"ContextVar Chroma 注入异常: {ctx_err}", exc_info=True)
        try:
            session_store.set_summary_dir(summary_dir)
        except Exception as ctx_err:
            logger.error(f"ContextVar SummaryDir 注入异常: {ctx_err}", exc_info=True)

        # 保存门禁：仅白名单提问放行
        session_store.set_save_allowed(save_requested)
        session_store.set_current_session_id(sid)
        if save_requested:
            logger.info(f"✅ 保存门禁已放行（检测到保存关键词）")
        else:
            logger.info(f"🔒 保存门禁已锁定（普通问答）")

        # ============ 步骤 1：先渲染用户提问 ============
        chat_history.append({"role": "user", "content": user_input})
        yield "", chat_history, session_state

        # ============ 步骤 2：渲染加载动画占位 ============
        LOADING_TEXT = "⏳ 正在检索文档、梳理答案中，请稍候…"
        chat_history.append({"role": "assistant", "content": LOADING_TEXT})
        yield "", chat_history, session_state

        # ============ 步骤 3：流式生成助手回答 ============
        loading_replaced = False
        try:
            graph_config = {"configurable": {"thread_id": sid}}
            async for event in agent_app.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                config=graph_config,
                version="v2"
            ):
                try:
                    kind = event.get("event")
                    if kind == "on_chat_model_stream":
                        chunk_data = event.get("data", {}).get("chunk")
                        if chunk_data and hasattr(chunk_data, "content"):
                            content = chunk_data.content
                            if content:
                                if not loading_replaced:
                                    chat_history[-1]["content"] = ""
                                    loading_replaced = True
                                chat_history[-1]["content"] += content
                                yield "", chat_history, session_state

                except Exception as chunk_err:
                    logger.warning(f"流式 chunk 异常，跳过: {chunk_err}")
                    continue

            # 回退：未捕获到任何 token → ainvoke 补充（异步非阻塞）
            if not loading_replaced:
                try:
                    result = await agent_app.ainvoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=graph_config,
                    )
                    answer = _extract_answer(result)
                    chat_history[-1]["content"] = answer
                except Exception as invoke_err:
                    logger.error(f"回退 ainvoke 异常: {invoke_err}", exc_info=True)
                    chat_history[-1]["content"] = "抱歉，系统处理超时，请缩短问题后重试。"

            # ============ 仅保存类提问：检查下载链接 ============
            if save_requested:
                try:
                    summary_path = os.path.join(summary_dir, "summary.txt")
                    if os.path.isfile(summary_path):
                        download_url = f"http://localhost:7863/api/download/{sid}"
                        download_link = (
                            f"\n\n---\n\n"
                            f"<a href='{download_url}' download "
                            f"style='display:inline-block;padding:8px 16px;background:#4CAF50;"
                            f"color:white;border-radius:4px;text-decoration:none;font-size:14px;'>"
                            f"📥 点击下载总结文件</a>"
                        )
                        chat_history[-1]["content"] += download_link
                        logger.info(f"下载链接已嵌入助手回复")
                except Exception:
                    pass

            yield "", chat_history, session_state

        except Exception as stream_err:
            logger.error(
                f"流式生成异常:\n{type(stream_err).__name__}: {stream_err}\n{traceback.format_exc()}"
            )
            chat_history[-1]["content"] = "抱歉，回答生成中断，请重试或换个问法。"
            yield "", chat_history, session_state

    except Exception as fatal_err:
        logger.error(
            f"对话执行致命异常:\n{type(fatal_err).__name__}: {fatal_err}\n{traceback.format_exc()}"
        )
        try:
            chat_history.append({"role": "user",
                                 "content": user_input if user_input else "（空输入）"})
            chat_history.append({"role": "assistant",
                                 "content": "抱歉，系统出现异常。请点击「清空全部对话」后重新提问。"})
        except Exception:
            chat_history = [
                {"role": "user", "content": user_input if user_input else "（空输入）"},
                {"role": "assistant",
                 "content": "抱歉，系统出现异常。请点击「清空全部对话」后重新提问。"},
            ]
        yield "", chat_history, session_state
        return

    finally:
        try:
            session_store.clear_current()
        except Exception:
            pass