import os
import sys
import json
import logging
import time
from typing import Dict, Any

# Configure logging


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from mlflow_utils import MLflowTracker, create_mlflow_run_tags
from logger import get_logger
import mlflow
logger=get_logger(__name__)

def initialize_inference_system(
    model_path: str = 'artifacts/models/churn_rf_model',
    encoders_path: str = 'artifacts/encode'
) -> ModelInference:
    """
    Initialize the inference system with model and encoders.
    """
    logger.info("Initializing inference system...")
    try:
        inference = ModelInference(model_path)
        logger.info("✓ Inference system initialized successfully")
        return inference
    except Exception as e:
        logger.error(f"✗ Failed to initialize inference system: {str(e)}")
        raise


def streaming_inference(inference: ModelInference, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform streaming inference with MLflow tracking.
    """
    mlflow_tracker = MLflowTracker()
    run_tags = create_mlflow_run_tags('streaming_inference', {
        'inference_type': 'single_record',
        'model_type': 'RandomForest'
    })
    run = mlflow_tracker.start_run(run_name='streaming_inference', tags=run_tags)
    
    try:
        logger.info("Processing inference request...")
        logger.info(f"Input keys: {list(data.keys())}")
        
        start_time = time.time()
        prediction_result = inference.predict(data)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'inference_time_ms': inference_time_ms,
            'churn_probability': float(prediction_result['confidence'].replace('%', '')) / 100,
            'predicted_class': 1 if prediction_result['prediction_label'] == 'Churn' else 0
        })
        mlflow.log_params({f'input_{k}': v for k, v in data.items()})
        
        logger.info("✓ Streaming inference completed")
        logger.info(f"Result: {prediction_result}, Inference time: {inference_time_ms:.2f} ms")
        return prediction_result
        
    except Exception as e:
        logger.error(f"✗ Streaming inference failed: {str(e)}")
        raise
    finally:
        mlflow_tracker.end_run()


# Example usage
if __name__ == "__main__":
    # Initialize inference system
    inference = initialize_inference_system()
    sample_data = {
    "customerID": "467-CHFZW",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 47,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 99.35,
    "TotalCharges": 100
}
    
    pred = streaming_inference(inference, sample_data)
    print(pred)
