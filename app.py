import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

from langchain_core.runnables import RunnableConfig

from langchain_agent import chat_agent
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ChatInput(BaseModel):
    input: Optional[str] = None


app = FastAPI()


@app.post("/chat")
async def chat(chat_input: ChatInput) -> StreamingResponse:
    # Check if a simple input string is provided
    system_prompt = mlflow.load_prompt("system_prompt")
    print(f"system_prompt: {system_prompt}")
    messages = [{"role": "system", "content": system_prompt.template}]

    if chat_input.input:
        # Create a user message from the input string and add it to messages
        user_message = {"role": "user", "content": chat_input.input}
        messages.append(user_message)

    async def generate():
        for event in chat_agent.predict_stream({"messages": messages}, ):
            yield json.dumps(event.model_dump()) + "\n"

    return StreamingResponse(generate(), media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
