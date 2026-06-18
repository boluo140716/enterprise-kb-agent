"""
对话路由：SSE 流式响应 + 非流式回退
"""
import json
import os
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from langchain_core.messages import HumanMessage

from api.models import ChatRequest, ChatResponse
from api.dependency import inject_session, cleanup_session
from agent.graph_builder import agent_app
from web.session_utils import _extract_answer
from settings import TEMP_SUMMARY_DIR
from log_config import logger
import session_store

router = APIRouter(prefix="/api", tags=["对话"])


def _is_save_request(user_input: str) -> bool:
    """检测是否需要保存/导出"""
    from prompts import SAVE_KEYWORDS
    return any(kw in user_input for kw in SAVE_KEYWORDS)


async def _stream_chat_events(user_input: str, session_id: str):
    """
    SSE 事件生成器，逐 token 流式推送。

    事件格式：
      data: {"type": "token", "content": "你"}
      data: {"type": "token", "content": "好"}
      ...
      data: {"type": "done", "session_id": "abc123"}
    """
    try:
        graph_config = {"configurable": {"thread_id": session_id}}
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
                            yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"
            except Exception:
                continue

        # 检测是否有保存内容
        download_url = None
        filepath = os.path.join(TEMP_SUMMARY_DIR, session_id, "summary.txt")
        if os.path.isfile(filepath):
            download_url = f"/api/download/{session_id}"

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'download_url': download_url}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"SSE 流式异常: {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': '回答生成中断，请重试'}, ensure_ascii=False)}\n\n"
    finally:
        cleanup_session()


@router.get("/download/{session_id}")
async def download_file(session_id: str):
    """
    下载保存的总结文件。
    浏览器收到 Content-Disposition: attachment 后会弹出"另存为"对话框。
    从文件读取（Gradio 和 FastAPI 不同进程，文件是唯一共享存储）。
    """
    filepath = os.path.join(TEMP_SUMMARY_DIR, session_id, "summary.txt")
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="没有可下载的内容，请先执行保存操作")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 下载后清理内存缓存
    session_store.remove_stored_summary(session_id)

    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename=summary_{session_id}.txt"
        },
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE 流式对话接口。

    客户端使用 EventSource 或 fetch + ReadableStream 接收：
    - 逐 token 实时推送，改善体感延迟
    - 连接断开自动重连
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    save_requested = _is_save_request(req.question)
    session_id, _ = inject_session(req.session_id, save_allowed=save_requested)

    return StreamingResponse(
        _stream_chat_events(req.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            "X-Session-Id": session_id,
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    非流式对话接口（一次性返回完整回答）。

    适用场景：批量调用、自动化脚本、不需要流式的前端
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    save_requested = _is_save_request(req.question)
    session_id, _ = inject_session(req.session_id, save_allowed=save_requested)

    try:
        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content=req.question)]},
            config={"configurable": {"thread_id": session_id}},
        )
        answer = _extract_answer(result)

        # 检测是否有保存内容，返回下载链接
        download_url = None
        filepath = os.path.join(TEMP_SUMMARY_DIR, session_id, "summary.txt")
        if os.path.isfile(filepath):
            download_url = f"/api/download/{session_id}"

        return ChatResponse(answer=answer, session_id=session_id, download_url=download_url)
    except Exception as e:
        logger.error(f"非流式对话异常: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"系统异常: {str(e)}")
    finally:
        cleanup_session()