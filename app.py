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
    try:
        # Set up MLflow experiment for tracking
        mlflow.set_experiment("api_service_metrics")

        # Start MLflow run for this request
        with mlflow.start_run(run_name="chat_request"):
            # Log the start time
            import time
            start_time = time.time()

            # Check if a simple input string is provided
            system_prompt = mlflow.load_prompt("system_prompt")
            print(f"system_prompt: {system_prompt}")
            messages = [{"role": "system", "content": system_prompt.template}]

            if chat_input.input:
                # Create a user message from the input string and add it to messages
                user_message = {"role": "user", "content": chat_input.input}
                messages.append(user_message)

                # Log input length as a metric
                mlflow.log_metric("input_length", len(chat_input.input))

            async def generate():
                try:
                    for event in chat_agent.predict_stream({"messages": messages}):
                        yield json.dumps(event.model_dump()) + "\n"
                except Exception as e:
                    # Log the error
                    mlflow.log_param("error", str(e))
                    # Re-raise the exception to be handled by the outer try-except
                    raise
                finally:
                    # Log the response time
                    end_time = time.time()
                    mlflow.log_metric("response_time_ms", (end_time - start_time) * 1000)

            return StreamingResponse(generate(), media_type="application/json")
    except Exception as e:
        # Log the error
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in chat endpoint: {error_details}")

        # Return a 500 error
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
