"""
LangGraph 所有业务节点：思考节点、工具执行节点、校验节点
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from settings import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_GPU_NUM
from prompts import SYS_PROMPT, JUDGE_PROMPT_TPL
from tools.agent_tools import tool_list

# 初始化LLM实例
llm = ChatOllama(
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    num_gpu=LLM_GPU_NUM,
    extra_kwargs={"numa": True}
)

def agent_think_node(state):
    """LLM思考节点：决定调用哪个工具"""
    full_messages = [HumanMessage(content=SYS_PROMPT)] + state["messages"]
    llm_with_tools = llm.bind_tools(tool_list)
    ai_msg = llm_with_tools.invoke(full_messages)
    return {"messages": [ai_msg]}

def tool_execute_node(state):
    """工具执行节点：执行具体工具逻辑"""
    last_msg = state["messages"][-1]
    tool_msg_collect = []
    for call_info in last_msg.tool_calls:
        target_tool = next(t for t in tool_list if t.name == call_info["name"])
        tool_result = target_tool.invoke(call_info["args"])
        tool_msg_collect.append(ToolMessage(
            content=tool_result,
            tool_call_id=call_info["id"]
        ))
    return {"messages": tool_msg_collect}

def judge_check_node(state):
    """结果校验节点：判断内容是否有效，决定是否重搜"""
    last_tool_msg = state["messages"][-1]
    # 提取用户原始问题
    user_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    # 执行校验
    judge_prompt = JUDGE_PROMPT_TPL.format(
        user_q=user_question,
        tool_content=last_tool_msg.content[:1200]
    )
    judge_res = llm.invoke(judge_prompt)
    return {"messages": [AIMessage(content=judge_res.content.strip())]}

def final_answer_node(state):
    """最终答案生成节点：基于检索结果，生成用户可读的自然语言回答"""
    # 提取用户原始问题（第一条HumanMessage）
    user_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # 提取最近一次工具返回的检索内容
    tool_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_content = msg.content
            break

    answer_prompt = f"""你是一个企业内部知识库助手。请基于以下检索到的文档内容，用中文专业、完整地回答用户问题。

用户问题：{user_question}

检索到的相关内容：
{tool_content[:3000]}

请直接给出清晰完整的答案，不要提及"根据检索结果"等套话。如果检索内容不足，请如实说明。"""

    answer = llm.invoke(answer_prompt)
    return {"messages": [AIMessage(content=answer.content)]}