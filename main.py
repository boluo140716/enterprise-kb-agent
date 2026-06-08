"""
项目主入口：控制台交互界面
"""
from agent.graph_builder import agent_app
from langchain_core.messages import HumanMessage
from log_config import logger
import traceback

def main():
    logger.info("企业内部知识库智能问答Agent 启动成功")
    print("功能：内部文档查询、行业资讯联网检索、文档总结导出")
    print("输入 exit / quit / 再见 退出程序\n")
    while True:
        user_input = input("员工提问：")
        exit_words = ["exit", "quit", "再见"]
        if user_input.lower() in exit_words:
            logger.info("程序正常退出")
            print("程序退出，感谢使用！")
            break

        try:
            resp_state = agent_app.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
            final_answer = resp_state["messages"][-1].content
            print(f"知识库助手：{final_answer}\n")
        except Exception as e:
            err_stack = traceback.format_exc()
            logger.error(f"对话执行异常详情：\n异常信息：{str(e)}\n完整堆栈：\n{err_stack}")
            print("系统异常，请重新提问\n")

if __name__ == "__main__":
    main()