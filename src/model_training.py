import os
import sys
import time
import json
from typing import Optional, List, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from spark_session import get_or_create_spark_session

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger
logger = get_logger(__name__)
class ModelTrainer:
    """
    Enhanced PySpark model trainer with Pipeline support for consistent feature assembly.
    """

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize the model trainer with a Spark session."""
        self.spark = spark or get_or_create_spark_session()
        logger.info("ModelTrainer initialized")

    def train(self, model, df: DataFrame, featureCols: List[str], labelCol: str) -> Tuple:
        try:
            df_renamed = df.withColumnRenamed(labelCol, "label")
            logger.info(f"Training samples: {df_renamed.count()}")
            logger.info(f"Features: {len(featureCols)}")
            logger.info(f"Feature columns: {featureCols}")

            # Save feature column names for inference
            feature_cols_mapping = {i: col for i, col in enumerate(featureCols)}
            feature_columns_path = os.path.join('artifacts','feature_columns.json')
            os.makedirs(os.path.dirname(feature_columns_path), exist_ok=True)
            with open(feature_columns_path, 'w') as f:
                json.dump(feature_cols_mapping, f, indent=2)
            logger.info(f"✓ Saved feature columns for inference → {feature_columns_path}")

            sample_rows = df.select(featureCols).limit(5).toPandas()
            logger.info(f"Sample rows:\n{sample_rows}")


            # Create feature assembler
            assembler = VectorAssembler(
                inputCols=featureCols,
                outputCol="features",
                handleInvalid="skip"
            )

            # Create pipeline
            pipeline = Pipeline(stages=[assembler, model])
            logger.info("Pipeline created: [VectorAssembler, Model]")
            
            start_time = time.time()
            trained_pipeline_model = pipeline.fit(df_renamed)
            training_time = time.time() - start_time
            logger.info(f"✓ Pipeline training completed in {training_time:.2f} seconds")
            logger.info(f"{'='*60}\n")

            return trained_pipeline_model, training_time

        except Exception as e:
            logger.error(f"✗ Model training failed: {str(e)}")
            raise


    def save_model(self, model, save_path: str):
        """
        Save a trained PySpark pipeline model.
        
        Args:
            model: Trained pipeline model
            save_path (str): Path to save the model
        """
        if model is None:
            raise ValueError("No model to save.")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Use overwrite mode for PySpark ML models
            model.write().overwrite().save(save_path)
            
            logger.info(f"✓ Pipeline model saved successfully at {save_path}")
        except Exception as e:
            logger.error(f"✗ Failed to save model: {str(e)}")
            raise

    def load_model(self, load_path: str):
        """
        Load a saved PySpark pipeline model.
        
        Args:
            load_path (str): Path to the saved model
            
        Returns:
            PipelineModel: Loaded pipeline model
        """
        if not os.path.exists(load_path):
            raise ValueError(f"Model path does not exist: {load_path}")
        try:
            from pyspark.ml import PipelineModel
            model = PipelineModel.load(load_path)
            logger.info(f"✓ Pipeline model loaded successfully from {load_path}")
            return model
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise