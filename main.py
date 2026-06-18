"""
项目主入口：控制台交互界面（支持流式输出）
"""
import asyncio
from agent.graph_builder import agent_app
from langchain_core.messages import HumanMessage, AIMessage
from log_config import logger
import traceback


def _extract_answer(result):
    """从 graph 最终状态提取可读答案，防御 tool_call 残留"""
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls and not last_msg.content:
        # 最后一条消息只有 tool_call 无文本 → 向前找文本回答
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                # 检查 tool_calls，确保没有未执行的工具调用
                if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                    return msg.content
        return "抱歉，未能生成有效回答，请重试。"
    return last_msg.content or "（空回答）"


async def main():
    logger.info("企业内部知识库智能问答Agent 启动成功")
    print("功能：内部文档查询、行业资讯联网检索、文档总结导出")
    print("输入 exit / quit / 再见 退出程序\n")

    while True:
        try:
            user_input = input("员工提问：")
        except (EOFError, KeyboardInterrupt):
            print("\n程序退出，感谢使用！")
            break

        exit_words = ["exit", "quit", "再见"]
        if user_input.lower() in exit_words:
            logger.info("程序正常退出")
            print("程序退出，感谢使用！")
            break

        try:
            print("知识库助手：", end="", flush=True)
            streamed = False

            # 流式输出：逐 token 打印，改善体感延迟
            # 使用固定 thread_id 实现控制台多轮对话上下文记忆
            graph_config = {"configurable": {"thread_id": "console"}}
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
                                print(content, end="", flush=True)
                                streamed = True
                except Exception:
                    continue

            if not streamed:
                # 回退：无流式 token（如纯工具调用后强制回答）
                result = await agent_app.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=graph_config,
                )
                answer = _extract_answer(result)
                print(answer)
            else:
                print()

        except Exception as e:
            err_stack = traceback.format_exc()
            logger.error(f"对话执行异常详情：\n异常信息：{str(e)}\n完整堆栈：\n{err_stack}")
            print(f"\n系统异常：{str(e)}\n")


if __name__ == "__main__":
    asyncio.run(main())