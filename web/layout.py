"""
Gradio UI 布局与事件绑定

兼容 Gradio 6.x，对话历史使用字典格式 [{"role":"user"/"assistant","content":"..."}, ...]
摘要作为 Markdown <details> 可折叠块嵌入助手回复，同豆包附件交互风格。
"""
import gradio as gr
from web.upload import handle_upload, clear_upload, clear_all
from web.chat import chat_respond


def _save_to_browser(chat_history, session_state):
    """将对话历史和会话 ID 保存到浏览器 localStorage"""
    return {
        "session_id": session_state.get("session_id", ""),
        "chat_history": chat_history,
    }


def _restore_from_browser(browser_state):
    """从浏览器 localStorage 恢复对话历史和会话 ID"""
    if browser_state and browser_state.get("chat_history"):
        return (
            browser_state["chat_history"],
            {"chroma": None, "file_names": [], "file_summaries": [],
             "session_id": browser_state.get("session_id", "")},
            browser_state,
        )
    return [], {"chroma": None, "file_names": [], "file_summaries": [], "session_id": ""}, browser_state


# ===================== 构建 Web 界面 =====================
with gr.Blocks(title="企业知识库RAG智能问答系统") as demo:
    # 会话私有状态：每个浏览器标签页独立
    session_state = gr.State({
        "chroma": None,
        "file_names": [],
        "file_summaries": [],
        "session_id": "",
    })

    # 浏览器持久化存储：刷新不丢失对话历史
    browser_state = gr.BrowserState({"session_id": "", "chat_history": []})

    gr.Markdown("""
    # 🏢 企业内部知识库智能问答Agent
    功能：内部文档检索、行业资讯联网查询、文档内容总结导出
    """)

    chat_box = gr.Chatbot(
        height=550,
        label="对话记录",
    )

    # 折叠上传面板
    with gr.Accordion("📁 上传本地文档（临时会话）", open=False):
        upload_file = gr.File(
            label="选择文件",
            file_types=[".pdf", ".docx", ".txt", ".md", ".xlsx", ".xls"],
            file_count="multiple",
        )
        with gr.Row():
            upload_btn = gr.Button("上传并索引", variant="secondary")
            clear_upload_btn = gr.Button("清空上传文件", variant="stop", size="sm")
        upload_status = gr.Markdown("📂 尚未上传临时文档")

    input_text = gr.Textbox(
        label="提问输入框",
        placeholder="请输入问题，例如：试用期离职需要提前多久告知？",
        lines=2,
    )
    with gr.Row():
        submit_btn = gr.Button("提交提问", variant="primary")
        clear_btn = gr.Button("清空全部对话")

    # ---- 事件绑定 ----
    # 上传 / 清空（同步函数，仅修改 session_state）
    upload_btn.click(
        fn=handle_upload,
        inputs=[upload_file, session_state],
        outputs=[session_state, upload_status],
    )

    clear_upload_btn.click(
        fn=clear_upload,
        inputs=[session_state],
        outputs=[session_state, upload_status],
    )

    # 对话（异步生成器，流式返回；concurrency_limit=1 串行执行，异常自动释放锁）
    chat_outputs = [input_text, chat_box, session_state]
    chat_event = submit_btn.click(
        fn=chat_respond,
        inputs=[input_text, chat_box, session_state],
        outputs=chat_outputs,
        concurrency_limit=1,
    )
    input_text.submit(
        fn=chat_respond,
        inputs=[input_text, chat_box, session_state],
        outputs=chat_outputs,
        concurrency_limit=1,
    ).then(
        fn=_save_to_browser,
        inputs=[chat_box, session_state],
        outputs=[browser_state],
    )
    # 对话完成后自动保存到浏览器 localStorage
    chat_event.then(
        fn=_save_to_browser,
        inputs=[chat_box, session_state],
        outputs=[browser_state],
    )

    # 页面加载时从浏览器 localStorage 恢复对话历史
    demo.load(
        fn=_restore_from_browser,
        inputs=[browser_state],
        outputs=[chat_box, session_state, browser_state],
    )

    # 清空全部：同步重置聊天记录 + 上传状态 + 输入框 + 浏览器缓存
    clear_btn.click(
        fn=clear_all,
        inputs=[session_state],
        outputs=[chat_box, session_state, upload_status, input_text],
    ).then(
        fn=lambda: {"session_id": "", "chat_history": []},
        inputs=[],
        outputs=[browser_state],
    )