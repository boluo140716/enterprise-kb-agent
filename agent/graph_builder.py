"""
图构建模块：整合状态、节点、路由，编译最终Agent
"""
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import agent_think_node, tool_execute_node, judge_check_node, final_answer_node
from agent.routes import tool_route_func, judge_route_func

def build_agent_graph():
    """构建并编译LangGraph图"""
    graph = StateGraph(AgentState)

    # 注册所有节点
    graph.add_node("agent_think_node", agent_think_node)
    graph.add_node("tool_execute_node", tool_execute_node)
    graph.add_node("judge_check_node", judge_check_node)
    graph.add_node("final_answer_node", final_answer_node)

    # 入口节点
    graph.set_entry_point("agent_think_node")

    # 配置边与条件路由
    graph.add_conditional_edges("agent_think_node", tool_route_func)
    graph.add_edge("tool_execute_node", "judge_check_node")
    graph.add_conditional_edges("judge_check_node", judge_route_func)
    graph.add_edge("final_answer_node", END)

    # 编译返回实例
    return graph.compile()

# 全局唯一Agent实例
agent_app = build_agent_graph()