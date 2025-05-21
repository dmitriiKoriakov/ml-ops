from typing import Optional, Sequence, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph, logger
from langgraph.prebuilt import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode


def create_tool_calling_agent(
        model: LanguageModelLike,
        tools: Union[ToolNode, Sequence[BaseTool]],
) -> CompiledGraph:
    model = model.bind_tools(tools)
    prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("messages")]
    )
    chain = prompt | model

    def routing_logic(state: ChatAgentState):
        last_message = state["messages"][-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    def call_model(
            state: ChatAgentState,
            config: RunnableConfig,
    ):
        response = chain.invoke({"messages": state["messages"]})

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        routing_logic,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
