"""
Agent 状态定义模块
"""
from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """智能体全局状态：消息列表自动累加"""
    messages: Annotated[Sequence[BaseMessage], operator.add]