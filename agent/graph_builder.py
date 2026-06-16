"""
图构建模块：ReAct 双节点图（think ⇄ tool_execute）
支持 MemorySaver 多轮对话上下文记忆
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes import agent_think_node, tool_execute_node
from agent.routes import tool_route_func


def build_agent_graph():
    """构建并编译 LangGraph ReAct 图（带内存检查点，支持多轮对话）"""
    graph = StateGraph(AgentState)

    graph.add_node("agent_think_node", agent_think_node)
    graph.add_node("tool_execute_node", tool_execute_node)

    graph.set_entry_point("agent_think_node")

    # think → (有 tool_calls?) → execute → think → (无 tool_calls?) → END
    graph.add_conditional_edges("agent_think_node", tool_route_func)
    graph.add_edge("tool_execute_node", "agent_think_node")

    return graph.compile(checkpointer=MemorySaver())


# 全局唯一 Agent 实例（支持多轮对话上下文记忆）
agent_app = build_agent_graph()
