"""
LangGraph 业务节点：ReAct 思考节点 + 工具执行节点
合并原 think / judge / final_answer 三节点为双节点循环
"""
import contextvars
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from settings import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_GPU_NUM, MAX_TOOL_ROUNDS
from prompts import SYS_PROMPT
from tools.agent_tools import tool_list
from log_config import logger

# 知识库检索结果缓存（ContextVar 实现，多标签页/多请求隔离）
# LangGraph 会丢弃 AgentState TypedDict 中未声明的 key，因此使用 ContextVar 旁路缓存
_kb_docs_cache = contextvars.ContextVar("kb_docs_cache", default="")

# 初始化 LLM 实例
# 注意：numa 是 Ollama 服务端配置（启动时 OLLAMA_NUMA=true），不是 chat API 参数
# 不要在 extra_kwargs 中传递，否则新版 ollama 客户端会拒绝
llm = ChatOllama(
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    num_gpu=LLM_GPU_NUM,
)

# 上下文截断上限（字符数），约 2500~3000 tokens，避免消息膨胀拖慢 CPU 推理
MAX_CONTEXT_CHARS = 10000


def _compress_tool_results(tool_msgs: list, max_chars: int) -> list:
    """
    多条 ToolMessage 超限时等比精简：每条保留首尾关键内容，绝不整条丢弃。
    """
    if not tool_msgs:
        return tool_msgs

    total = sum(len(str(m.content)) for m in tool_msgs)
    if total <= max_chars:
        return tool_msgs

    n = len(tool_msgs)
    per_msg = max(300, max_chars // n)  # 每条至少保留 300 字符

    compressed = []
    for msg in tool_msgs:
        content = str(msg.content)
        if len(content) <= per_msg:
            compressed.append(msg)
        else:
            half = per_msg // 2
            truncated = content[:half] + "\n...[内容过长已精简]...\n" + content[-half:]
            compressed.append(ToolMessage(
                content=truncated,
                tool_call_id=msg.tool_call_id
            ))
    return compressed


def _truncate_messages(messages: list, max_chars: int = MAX_CONTEXT_CHARS) -> list:
    """
    固定分段保留策略（不修改 state["docs"]）：
      系统提示词  →  永久保留
      用户提问(HumanMessage)  →  永久保留
      全部 ToolMessage 检索结果  →  永久保留
      AI 历史对话(AIMessage)  →  仅删减末尾部分

    保持时间顺序不变（不重排），仅从末尾逐条删除 AIMessage 直到不超过上限。
    知识库检索分片绝不丢弃；消息裁剪不影响文档可用性。
    """
    if len(messages) <= 2:
        return messages

    # 计算总字符数
    total = sum(len(str(m.content)) for m in messages)
    if total <= max_chars:
        return messages

    # 从末尾逐条删除 AIMessage（保留 SystemMsg / HumanMessage / ToolMessage）
    result = list(messages)
    while len(result) > 1:
        current_total = sum(len(str(m.content)) for m in result)
        if current_total <= max_chars:
            break

        # 从后往前找第一个可删除的 AIMessage（不是 system 提示词）
        dropped = False
        for i in range(len(result) - 1, 0, -1):
            if isinstance(result[i], AIMessage):
                result.pop(i)
                dropped = True
                break

        if not dropped:
            # 没有可删除的 AIMessage，压缩 ToolMessage 后退出
            break

    # 最终兜底：若仍超限，等比压缩 ToolMessage
    final_total = sum(len(str(m.content)) for m in result)
    if final_total > max_chars:
        tool_indices = [i for i, m in enumerate(result) if isinstance(m, ToolMessage)]
        if tool_indices:
            tool_chars = sum(len(str(result[i].content)) for i in tool_indices)
            other_chars = final_total - tool_chars
            budget = max(500, max_chars - other_chars)
            tool_msgs = [result[i] for i in tool_indices]
            compressed = _compress_tool_results(tool_msgs, budget)
            for idx, new_msg in zip(tool_indices, compressed):
                result[idx] = new_msg

    if len(result) < len(messages):
        logger.info(
            f"上下文截断: {len(messages)} → {len(result)} 条消息 (约 {sum(len(str(m.content)) for m in result)} 字符)"
        )
    return result


def agent_think_node(state):
    """
    ReAct 思考节点：LLM 决定调用工具或直接输出最终答案。

    工具绑定策略：
    - 知识库已检索到有效文档 → 移除 search_knowledge_base，禁止重复检索浪费轮数
    - 工具调用未达上限 → 绑定工具，LLM 可自由选择调工具或直接回答
    - 已达上限 → 不绑定工具，强制 LLM 输出纯文本答案
    """
    docs_cache = _kb_docs_cache.get()

    # 回退：若 ContextVar 未命中（LangGraph 可能隔离节点上下文），
    # 从 state["messages"] 中提取已有的 KB 检索结果
    if not docs_cache:
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage) and msg.content and len(msg.content.strip()) > 100:
                if not msg.content.startswith("[工具异常]") and not msg.content.startswith("[系统错误]"):
                    docs_cache = msg.content
                    _kb_docs_cache.set(docs_cache)
                    break

    # 注入缓存的知识库文档（不受消息裁剪影响），确保 LLM 始终可见
    if docs_cache:
        enhanced_prompt = (
            SYS_PROMPT
            + "\n\n[已检索到的知识库文档——必须严格基于以下内容回答，禁止输出工具调用引导文本]\n"
            + docs_cache
        )
        raw_messages = [HumanMessage(content=enhanced_prompt)] + state["messages"]
    else:
        raw_messages = [HumanMessage(content=SYS_PROMPT)] + state["messages"]

    messages = _truncate_messages(raw_messages)

    # 统计已执行工具轮数，判断是否需要强制文本回答
    tool_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    force_answer = tool_count >= MAX_TOOL_ROUNDS

    # 知识库去重：已检索到有效文档 → 强制 LLM 基于文档直接回答，不绑定任何工具
    # qwen2:7b 在绑定工具时即使已知答案也倾向输出空内容，故直接切到纯文本模式
    if docs_cache:
        force_answer = True
        logger.info("知识库已缓存有效文档，跳过工具绑定，强制 LLM 基于文档直接回答")

    if force_answer:
        # 不绑定工具，强制 LLM 给出最终文本回答
        logger.info(f"强制纯文本回答模式 (工具轮数 {tool_count}/{MAX_TOOL_ROUNDS})")
        ai_msg = llm.invoke(messages)
    else:
        # 未达上限：绑定全部工具，LLM 自行决定调用工具或直接回答
        llm_with_tools = llm.bind_tools(tool_list)
        ai_msg = llm_with_tools.invoke(messages)

    return {"messages": [ai_msg]}


def tool_execute_node(state):
    """
    工具执行节点：执行 LLM 指定的工具，返回 ToolMessage。
    异常与空结果均保留可读标记，方便 LLM 判断下一步动作。

    search_knowledge_base 返回的有效文档会缓存到 ContextVar，
    后续消息裁剪不影响文档可用性，LLM 每轮都能看到检索结果。
    """
    last_msg = state["messages"][-1]
    tool_msg_collect = []
    docs_cache = _kb_docs_cache.get()

    for call_info in last_msg.tool_calls:
        tool_name = call_info.get("name", "")
        try:
            target_tool = next(t for t in tool_list if t.name == tool_name)
            tool_result = target_tool.invoke(call_info["args"])
            content = str(tool_result) if tool_result is not None else ""
        except StopIteration:
            logger.error(f"未找到工具: {tool_name}")
            content = f"[系统错误] 未找到工具: {tool_name}"
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行异常: {e}", exc_info=True)
            content = f"[工具异常] {type(e).__name__}: {str(e)}"

        # 知识库检索结果缓存到 ContextVar，不受 LangGraph 状态裁剪影响
        # 仅缓存有效文档内容，异常/空结果不缓存，保留重试机会
        if tool_name == "search_knowledge_base" and content and not docs_cache:
            if not content.startswith("[工具异常]") and not content.startswith("[系统错误]"):
                _kb_docs_cache.set(content)
                logger.info(f"知识库检索结果已缓存至 ContextVar ({len(content)} 字符)")

        tool_msg_collect.append(ToolMessage(
            content=content,
            tool_call_id=call_info["id"]
        ))

    return {"messages": tool_msg_collect}
