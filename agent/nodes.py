"""
LangGraph 业务节点：ReAct 思考节点 + 工具执行节点
合并原 think / judge / final_answer 三节点为双节点循环
"""
import contextvars
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from prompts import SYS_PROMPT
from tools.agent_tools import tool_list
from log_config import logger
from settings import LLM_MODEL_NAME, LLM_TEMPERATURE,MAX_TOOL_ROUNDS, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
# 知识库检索结果缓存（ContextVar 实现，多标签页/多请求隔离）
# LangGraph 会丢弃 AgentState TypedDict 中未声明的 key，因此使用 ContextVar 旁路缓存
_kb_docs_cache = contextvars.ContextVar("kb_docs_cache", default="")

# LLM 实例（懒加载，避免 import 时阻塞）
_llm = None


def _get_llm():
    """获取 LLM 实例（懒加载单例）"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _llm

# 上下文截断上限（字符数），约 2500~3000 tokens，避免消息膨胀拖慢 CPU 推理
MAX_CONTEXT_CHARS = 20000


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
      AI 历史对话(AIMessage)  →  永久保留
      检索结果(ToolMessage)  →  优先删减（新问题会重新检索，旧结果价值最低）

    保持时间顺序不变（不重排），从末尾逐条删除 ToolMessage 直到不超过上限。
    """
    if len(messages) <= 2:
        return messages

    # 计算总字符数
    total = sum(len(str(m.content)) for m in messages)
    if total <= max_chars:
        return messages

    # 从末尾逐条删除 ToolMessage（保留 SystemMsg / HumanMessage / AIMessage）
    result = list(messages)
    while len(result) > 1:
        current_total = sum(len(str(m.content)) for m in result)
        if current_total <= max_chars:
            break

        # 从后往前找第一个可删除的 ToolMessage
        dropped = False
        for i in range(len(result) - 1, 0, -1):
            if isinstance(result[i], ToolMessage):
                result.pop(i)
                dropped = True
                break

        if not dropped:
            # 没有可删除的 ToolMessage，压缩 AIMessage 后退出
            break

    # 最终兜底：若仍超限，等比压缩 AIMessage
    final_total = sum(len(str(m.content)) for m in result)
    if final_total > max_chars:
        ai_indices = [i for i, m in enumerate(result) if isinstance(m, AIMessage)]
        if ai_indices:
            tool_chars = sum(len(str(result[i].content)) for i in ai_indices)
            other_chars = final_total - tool_chars
            budget = max(500, max_chars - other_chars)
            ai_msgs = [result[i] for i in ai_indices]
            compressed = _compress_tool_results(ai_msgs, budget)
            for idx, new_msg in zip(ai_indices, compressed):
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
    # 从 state["messages"] 中提取已有的 search_knowledge_base 检索结果
    if not docs_cache:
        for i, msg in enumerate(state.get("messages", [])):
            if isinstance(msg, ToolMessage) and msg.content and len(msg.content.strip()) > 10:
                if not msg.content.startswith("[工具异常]") and not msg.content.startswith("[系统错误]"):
                    # 校验消息来自 search_knowledge_base（非 search_online 等）
                    # 向前查找对应的 AIMessage tool_call 确认工具名称
                    for j in range(i - 1, -1, -1):
                        prev = state["messages"][j]
                        if hasattr(prev, "tool_calls"):
                            for tc in prev.tool_calls:
                                if tc.get("id") == msg.tool_call_id and tc.get("name") == "search_knowledge_base":
                                    docs_cache = msg.content
                                    _kb_docs_cache.set(docs_cache)
                                    break
                            if docs_cache:
                                break
                    if docs_cache:
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

    # 消除系统提示词重复注入：多轮 ReAct + 多轮对话中只保留最新一份 SYS_PROMPT
    # 多轮对话场景：不同轮次可能检索到不同 KB 文档，保留最新的 enhanced_prompt 确保 LLM 看到最新上下文
    deduped = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage) and str(msg.content).startswith(SYS_PROMPT[:50]):
            # 找到之前的系统提示词位置，移除旧版，后续追加新版
            for j in range(len(deduped) - 1, -1, -1):
                if isinstance(deduped[j], HumanMessage) and str(deduped[j].content).startswith(SYS_PROMPT[:50]):
                    deduped.pop(j)
                    break
        deduped.append(msg)
    raw_messages = deduped

    messages = _truncate_messages(raw_messages)

    # 统计已执行工具轮数，判断是否需要强制文本回答
    tool_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    force_answer = tool_count >= MAX_TOOL_ROUNDS

    # 知识库去重：KB 工具始终可用（多轮对话中每轮可能需要检索不同信息）
    # 仅当 tool_count 达到上限时才绑定空工具列表强制文本回答
    # LRU 缓存在 retriever.py 处理重复查询的性能优化
    available_tools = list(tool_list)

    llm = _get_llm()

    if force_answer:
        # 已达工具上限：不绑定工具，强制 LLM 给出最终文本回答
        logger.info(f"强制纯文本回答模式 (工具轮数 {tool_count}/{MAX_TOOL_ROUNDS})")
        ai_msg = llm.invoke(messages)
    else:
        # 未达上限：绑定工具，LLM 自行决定调用工具或直接回答
        llm_with_tools = llm.bind_tools(available_tools)
        ai_msg = llm_with_tools.invoke(messages)

        # qwen2:7b 兼容：有文档缓存但绑了工具时可能输出空内容
        # 此时退回到纯文本模式重试一次
        if docs_cache and not ai_msg.content and not (hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls):
            logger.info("工具绑定模式下 KB 问答输出为空，回退到纯文本模式")
            ai_msg = llm.invoke(messages)

    # 清理：有 KB 缓存但 LLM 仍输出 <tool_call> 伪文本时，重新提取答案
    # qwen2:7b 在纯文本模式下仍可能输出 tool_call XML 字符串（模型训练遗留行为）
    if docs_cache and ai_msg.content and "<tool_call>" in str(ai_msg.content):
        import re
        stripped = re.sub(r'<tool_call>.*?</tool_call>', '', str(ai_msg.content), flags=re.DOTALL).strip()
        if stripped:
            ai_msg = AIMessage(content=stripped)
        else:
            logger.info("<tool_call> 伪文本无有效内容，重试纯文本回答")
            ai_msg = llm.invoke(messages)
            # 二次回退：直接移除残留的 tool_call 文本
            if "<tool_call>" in str(ai_msg.content):
                ai_msg = AIMessage(content=re.sub(
                    r'<tool_call>.*?</tool_call>', '', str(ai_msg.content), flags=re.DOTALL
                ).strip() or "根据已检索到的企业制度文档，无法直接回答该问题。请尝试更具体的提问方式。")

    # 假保存检测：LLM 输出"复制保存"等文本但没调工具 → 丢弃、追加提醒、强制重试
    if not force_answer:
        content = getattr(ai_msg, "content", "") or ""
        has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
        fake_patterns = ["复制保存", "复制以下", "请复制", "由于系统限制", "无法直接通过工具", "无法直接保存"]
        if not has_tool_calls and any(p in content for p in fake_patterns):
            logger.warning("检测到 LLM 假装保存（未调工具），追加提醒并强制重试")
            messages.append(HumanMessage(
                content="【系统强制提醒】你刚才没有调用 save_summary_to_txt 工具！这严重违反了规则。"
                        "请立即调用 save_summary_to_txt 工具，将完整总结内容作为 summary_text 参数传入。"
                        "禁止在回答中直接输出文本。"
            ))
            llm_with_tools = llm.bind_tools(available_tools)
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
            tool_call_id=call_info.get("id", "")
        ))

    return {"messages": tool_msg_collect}