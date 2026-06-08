"""
路由判断函数：控制图的分支走向
"""
from langgraph.graph import END
from langchain_core.messages import ToolMessage

def tool_route_func(state) -> str:
    """判断是否需要执行工具"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_execute_node"
    return END

def judge_route_func(state) -> str:
    """校验后路由：ng→重试思考 / ok→生成最终答案 / 超过重试上限→强制生成答案"""
    judge_result = state["messages"][-1].content.strip()
    # 统计已执行工具次数，防止无限循环（最多3次重试）
    tool_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    if judge_result == "ng" and tool_count < 3:
        return "agent_think_node"
    # ok 或已达重试上限 → 生成最终答案
    return "final_answer_node"