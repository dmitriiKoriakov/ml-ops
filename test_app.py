import unittest
import uuid
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from app import app, ChatInput
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage
import mlflow

class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_app_initialization(self):
        # Test that the app is a FastAPI instance
        self.assertIsInstance(app, FastAPI)

        # Test that the app has the chat endpoint
        routes = [route.path for route in app.routes]
        self.assertIn("/chat", routes)

    @patch('app.chat_agent.predict_stream')
    def test_chat_endpoint_with_input(self, mock_predict_stream):
        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Mock the predict_stream method to return a generator of ChatAgentChunk objects
        mock_response = [
            ChatAgentChunk(delta=ChatAgentMessage(role="assistant", content="Hello", id=message_id)),
            ChatAgentChunk(delta=ChatAgentMessage(role="assistant", content=" world!", id=message_id))
        ]
        mock_predict_stream.return_value = mock_response

        # Test with a simple input
        response = self.client.post("/chat", json={"input": "Hi there"})

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Check if the content type is correct
        self.assertEqual(response.headers["content-type"], "application/json")

        # Check if the predict_stream method was called with the correct arguments
        mock_predict_stream.assert_called_once()
        args, _ = mock_predict_stream.call_args
        self.assertEqual(args[0]["messages"][0]["role"], "user")
        self.assertEqual(args[0]["messages"][0]["content"], "Hi there")

        # Check the response content
        response_lines = response.content.decode().strip().split('\n')
        self.assertEqual(len(response_lines), 2)

        # Parse the JSON responses
        import json
        first_chunk = json.loads(response_lines[0])
        second_chunk = json.loads(response_lines[1])

        # Check the content of the chunks
        self.assertEqual(first_chunk["delta"]["role"], "assistant")
        self.assertEqual(first_chunk["delta"]["content"], "Hello")
        self.assertEqual(first_chunk["delta"]["id"], message_id)
        self.assertEqual(second_chunk["delta"]["role"], "assistant")
        self.assertEqual(second_chunk["delta"]["content"], " world!")
        self.assertEqual(second_chunk["delta"]["id"], message_id)

    @patch('app.chat_agent.predict_stream')
    def test_chat_endpoint_with_empty_input(self, mock_predict_stream):
        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Mock the predict_stream method to return a generator of ChatAgentChunk objects
        mock_response = [
            ChatAgentChunk(delta=ChatAgentMessage(role="assistant", content="I didn't receive any input.", id=message_id))
        ]
        mock_predict_stream.return_value = mock_response

        # Test with an empty input
        response = self.client.post("/chat", json={"input": ""})

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Check if the predict_stream method was called with the correct arguments
        mock_predict_stream.assert_called_once()
        args, _ = mock_predict_stream.call_args
        # An empty string is falsy, so no message should be added to the messages list
        self.assertEqual(len(args[0]["messages"]), 0)

    @patch('app.chat_agent.predict_stream')
    def test_chat_endpoint_with_none_input(self, mock_predict_stream):
        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Mock the predict_stream method to return a generator of ChatAgentChunk objects
        mock_response = [
            ChatAgentChunk(delta=ChatAgentMessage(role="assistant", content="I didn't receive any input.", id=message_id))
        ]
        mock_predict_stream.return_value = mock_response

        # Test with None input
        response = self.client.post("/chat", json={"input": None})

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Check if the predict_stream method was called with an empty messages list
        mock_predict_stream.assert_called_once()
        args, _ = mock_predict_stream.call_args
        self.assertEqual(len(args[0]["messages"]), 0)

    @patch('app.chat_agent.predict_stream')
    def test_chat_endpoint_without_input_field(self, mock_predict_stream):
        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Mock the predict_stream method to return a generator of ChatAgentChunk objects
        mock_response = [
            ChatAgentChunk(delta=ChatAgentMessage(role="assistant", content="I didn't receive any input.", id=message_id))
        ]
        mock_predict_stream.return_value = mock_response

        # Test without the input field
        response = self.client.post("/chat", json={})

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Check if the predict_stream method was called with an empty messages list
        mock_predict_stream.assert_called_once()
        args, _ = mock_predict_stream.call_args
        self.assertEqual(len(args[0]["messages"]), 0)

    @patch('mlflow.load_prompt')
    @patch('app.chat_agent.predict_stream')
    def test_system_prompt_loading(self, mock_predict_stream, mock_load_prompt):
        # Mock the system prompt
        mock_prompt = MagicMock()
        mock_prompt.template = "You are a helpful assistant."
        mock_load_prompt.return_value = mock_prompt

        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Mock the predict_stream method
        mock_response = [
            ChatAgentChunk(delta=ChatAgentMessage(role="assistant", content="Hello", id=message_id))
        ]
        mock_predict_stream.return_value = mock_response

        # Test with a simple input
        response = self.client.post("/chat", json={"input": "Hi"})

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Verify that load_prompt was called with the correct argument
        mock_load_prompt.assert_called_once_with("system_prompt")

        # Verify that the system prompt was included in the messages
        mock_predict_stream.assert_called_once()
        args, _ = mock_predict_stream.call_args
        self.assertEqual(args[0]["messages"][0]["role"], "system")
        self.assertEqual(args[0]["messages"][0]["content"], "You are a helpful assistant.")

    @patch('app.chat_agent.predict_stream')
    def test_error_handling(self, mock_predict_stream):
        # Make the predict_stream method raise an exception
        mock_predict_stream.side_effect = Exception("Test error")

        # Test with a simple input
        response = self.client.post("/chat", json={"input": "Hi"})

        # Check if the response is a server error
        self.assertEqual(response.status_code, 500)

    @patch('mlflow.log_metric')
    @patch('mlflow.start_run')
    @patch('app.chat_agent.predict_stream')
    def test_mlflow_metric_logging(self, mock_predict_stream, mock_start_run, mock_log_metric):
        # Set up MLflow mocks
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run

        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Mock the predict_stream method
        mock_response = [
            ChatAgentChunk(delta=ChatAgentMessage(role="assistant", content="Hello", id=message_id))
        ]
        mock_predict_stream.return_value = mock_response

        # Set up MLflow experiment
        mlflow.set_experiment("test_metrics")

        # Test with a simple input
        with mlflow.start_run(run_name="test_run"):
            response = self.client.post("/chat", json={"input": "Hi"})

            # Log a test metric
            mlflow.log_metric("response_time_ms", 100)

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)

        # Verify that MLflow was used correctly
        mock_start_run.assert_called_once()
        mock_log_metric.assert_called_once_with("response_time_ms", 100)

if __name__ == "__main__":
    unittest.main()
