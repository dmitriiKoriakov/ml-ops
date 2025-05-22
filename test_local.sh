#!/bin/bash

# Exit on error
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tests with coverage..."
pytest test_app.py --cov=. --cov-report=xml

echo "Calculating MLflow metrics..."
python - << 'EOF'
import mlflow
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app

# Set up MLflow experiment
mlflow.set_experiment("api_service_metrics")

# Run a test with MLflow tracking
with mlflow.start_run(run_name="api_service_test"):
    # Log parameters
    mlflow.log_param("test_type", "unit_test")
    
    # Run tests and collect metrics
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromName('test_app.TestApp')
    test_result = unittest.TextTestRunner().run(test_suite)
    
    # Log metrics
    mlflow.log_metric("tests_run", test_result.testsRun)
    mlflow.log_metric("tests_passed", test_result.testsRun - len(test_result.errors) - len(test_result.failures))
    mlflow.log_metric("tests_failed", len(test_result.failures))
    mlflow.log_metric("tests_errored", len(test_result.errors))
    
    # Calculate and log success rate
    success_rate = (test_result.testsRun - len(test_result.errors) - len(test_result.failures)) / test_result.testsRun * 100
    mlflow.log_metric("success_rate", success_rate)
    
    print(f"Tests run: {test_result.testsRun}")
    print(f"Tests passed: {test_result.testsRun - len(test_result.errors) - len(test_result.failures)}")
    print(f"Tests failed: {len(test_result.failures)}")
    print(f"Tests errored: {len(test_result.errors)}")
    print(f"Success rate: {success_rate}%")
EOF

echo "All tests passed!"