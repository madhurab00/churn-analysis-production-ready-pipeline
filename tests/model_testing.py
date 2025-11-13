import os
import sys
from pyspark.sql import SparkSession


# Add utils and src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference

def debug_inference_test(model_path: str, artifacts_dir: str):
    """
    Test model inference with full preprocessing steps on multiple sample rows.
    This will help debug identical prediction issues.
    """
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("DebugInferenceTest") \
        .getOrCreate()

    # Initialize ModelInference
    mi = ModelInference(model_path=model_path, artifacts_dir=artifacts_dir, spark=spark)

    # Sample test rows (replace/add with actual rows from your dataset)
    test_rows = [
        {
            "customerID": "5248-YGIJN",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 72,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 90.25,
            "TotalCharges": 6369.45,
        },
        {
            "customerID": "1234-ABCD",
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "Yes",
            "tenure": 5,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 75.5,
            "TotalCharges": 378.5,
        },
    ]

    print("\n=== DEBUG INFERENCE TEST ===\n")
    for i, row in enumerate(test_rows):
        print(f"--- Row {i+1} ---")
        try:
            result = mi.predict(row)
            print(f"Prediction result: {result}\n")
        except Exception as e:
            print(f"Error processing row {i+1}: {e}\n")

    spark.stop()

if __name__ == "__main__":
    MODEL_PATH = "artifacts/models/churn_rf_model"   # Update path if different
    ENCODERS_PATH = "artifacts/encode"               # Update path if different
    debug_inference_test(MODEL_PATH, ENCODERS_PATH)
