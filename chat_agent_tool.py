import os

import mlflow
from langchain_openai import ChatOpenAI

from random import randint
from typing import Any

from langchain_core.tools import tool

from chat_agent import create_tool_calling_agent

# Set a specific experiment for MLflow tracing
mlflow.set_experiment("langchain_traces")


@tool
def generate_random_ints(min: int, max: int, size: int) -> dict[str, Any]:
    """Generate size random ints in the range [min, max]."""
    attachments = {"min": min, "max": max}
    custom_outputs = [randint(min, max) for _ in range(size)]
    content = f"Successfully generated array of {size} random ints in [{min}, {max}]."
    return {
        "content": content,
        "attachments": attachments,
        "custom_outputs": {"random_nums": custom_outputs},
    }


mlflow.langchain.autolog()
tools = [generate_random_ints]

llm = ChatOpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    model_name="qwen3:8b",
    temperature=0.7,
    api_key="ollama",
)
langgraph_agent = create_tool_calling_agent(llm, tools)
