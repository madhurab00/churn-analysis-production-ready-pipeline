import os
import sys
import json
import time
from typing import Dict, Any, Optional

import pandas as pd
from pyspark.sql import SparkSession

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from spark_session import create_spark_session, stop_spark_session
from data_pipeline import data_pipeline
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import RandomForestModelBuilder
from config import get_model_config, get_data_paths
from logger import get_logger
import mlflow
from mlflow_utils import MLflowTracker, create_mlflow_run_tags

logger = get_logger(__name__)

def training_pipeline(
        data_path: str = 'data/raw/churndataset.csv',
        model_params: Optional[Dict[str, Any]] = None,
        model_path: str = 'artifacts/models/churn_rf_model',
        data_format: str = 'csv'
):
    logger.info(f"\n{'='*80}")
    logger.info("STARTING MACHINE LEARNING TRAINING PIPELINE")
    logger.info(f"{'='*80}")

    # Run data pipeline first
    processed_data = data_pipeline(output_format=data_format)
    
    # Initialize Spark session
    spark = create_spark_session("ChurnPredictionTrainingPipeline")

    try:
        # Initialize MLflow tracker
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags('training_pipeline', {
            'model_type': 'RandomForest',
            'data_path': data_path,
            'model_path': model_path,
            'data_format': data_format,
            'processing_engine': 'pyspark'
        })
        run = mlflow_tracker.start_run(run_name='training_pipeline', tags=run_tags)

        # Convert numpy arrays to Spark DataFrame
        train_df = spark.createDataFrame(pd.DataFrame(processed_data['train']))
        test_df = spark.createDataFrame(pd.DataFrame(processed_data['test']))
        
        feature_cols = train_df.columns[:-1]
        label_col = train_df.columns[-1]

        # Build model
        model_builder = RandomForestModelBuilder(**(model_params or {}))
        model = model_builder.build_model()

        # Train model
        trainer = ModelTrainer(spark=spark)
        trained_model, training_time = trainer.train(model, train_df, feature_cols, label_col)

        # Save model locally
        trainer.save_model(trained_model, model_path)

        # Evaluate model
        evaluator = ModelEvaluator(trained_model, "RandomForest", spark=spark)
        evaluation_results = evaluator.evaluate(test_df, labelCol=label_col)

        # Log training metrics & model using MLflowTracker
        mlflow_tracker.log_training_metrics(
            model=trained_model,
            training_metrics={'training_time_seconds': training_time},
            model_params=model_params or {}
        )

        # Log evaluation metrics using MLflowTracker
        mlflow_tracker.log_evaluation_metrics(
            evaluation_metrics={'metrics': evaluation_results}
        )

        # Save summary artifact
        summary = {
            'model_type': 'RandomForest',
            'training_samples': train_df.count(),
            'test_samples': test_df.count(),
            'features_used': len(feature_cols),
            'training_time_seconds': training_time,
            'evaluation': evaluation_results
        }
        summary_path = os.path.join('artifacts','summary', f'training_summary_{int(time.time())}.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(summary_path)

        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        mlflow_tracker.end_run()

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        if 'mlflow_tracker' in locals():
            mlflow_tracker.end_run()
        raise
    finally:
        stop_spark_session(spark)


if __name__ == '__main__':
    model_config = get_model_config()
    training_pipeline(model_params=model_config.get('model_params'))
