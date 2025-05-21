from typing import Any, Generator, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from chat_agent_tool import langgraph_agent


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
            self,
            messages: list[ChatAgentMessage],
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        print(f"custom_inputs: {custom_inputs}")
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
            self,
            messages: list[ChatAgentMessage],
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        print(f"custom_inputs: {custom_inputs}")
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates", ):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


chat_agent = LangGraphChatAgent(langgraph_agent)
