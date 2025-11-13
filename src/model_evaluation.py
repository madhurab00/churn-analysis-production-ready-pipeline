import os
import sys
from typing import Dict, Any, Optional, List
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from spark_session import get_or_create_spark_session
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger
logger = get_logger(__name__)


class ModelEvaluator:
    """
    Class for evaluating PySpark ML models (including Pipeline models) using standard classification metrics.

    Attributes
    ----------
    model : object
        Trained PySpark ML model or PipelineModel with a transform() method.
    model_name : str
        Name of the model for logging purposes.
    spark : SparkSession
        Active Spark session.
    evaluation_results : dict
        Dictionary storing evaluation metrics after evaluation.
    """

    def __init__(self, model: Any, model_name: str, spark: Optional[SparkSession] = None):
        """
        Initialize the ModelEvaluator with a trained PySpark model.

        Args:
            model (Any): Trained PySpark ML model or PipelineModel.
            model_name (str): Name of the model for logging purposes.
            spark (SparkSession, optional): Spark session. If None, a new session is created.
        """
        self.model = model
        self.model_name = model_name
        self.spark = spark or get_or_create_spark_session()
        self.evaluation_results: Dict[str, Any] = {}

    def evaluate(self, df: DataFrame, labelCol: str = "label", predictionCol: str = "prediction") -> Dict[str, Any]:
        """
        Evaluate the model on a PySpark DataFrame and compute standard metrics.
        
        Note: If the model is a PipelineModel with VectorAssembler, it will automatically
        handle feature assembly. Pass the DataFrame with original column names.

        Args:
            df (DataFrame): Input DataFrame containing original columns (features and label).
            labelCol (str): Name of the label column in df.
            predictionCol (str): Name of the prediction column to be generated.

        Returns:
            dict: Dictionary containing accuracy, precision, recall, and F1 score.

        Raises:
            ValueError: If prediction fails or input data is invalid.
            Exception: For any other unexpected errors.
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"MODEL EVALUATION - {self.model_name}")
            logger.info(f"{'='*60}")
            
            # Rename label column to 'label' for evaluation
            df_renamed = df.withColumnRenamed(labelCol, "label")
            logger.info(f"Test samples: {df_renamed.count()}")

            # Generate predictions
            # If model is a PipelineModel, it will handle feature assembly automatically
            logger.info("Generating predictions...")
            predictions = self.model.transform(df_renamed)
            logger.info("✓ Predictions generated successfully")

            # Define evaluators
            accuracy_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol=predictionCol, metricName="accuracy"
            )
            f1_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol=predictionCol, metricName="f1"
            )
            precision_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol=predictionCol, metricName="weightedPrecision"
            )
            recall_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol=predictionCol, metricName="weightedRecall"
            )

            # Compute metrics
            logger.info("Computing evaluation metrics...")
            accuracy = accuracy_evaluator.evaluate(predictions)
            f1 = f1_evaluator.evaluate(predictions)
            precision = precision_evaluator.evaluate(predictions)
            recall = recall_evaluator.evaluate(predictions)

            self.evaluation_results = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"EVALUATION RESULTS - {self.model_name}")
            logger.info(f"{'='*60}")
            logger.info(f"✓ Accuracy:  {accuracy:.4f}")
            logger.info(f"✓ F1 Score:  {f1:.4f}")
            logger.info(f"✓ Precision: {precision:.4f}")
            logger.info(f"✓ Recall:    {recall:.4f}")
            logger.info(f"{'='*60}\n")

            return self.evaluation_results

        except ValueError as ve:
            logger.error(f"ValueError during evaluation of model {self.model_name}: {ve}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during evaluation of model {self.model_name}: {e}")
            raise

    def get_results(self) -> Dict[str, Any]:
        """
        Get the stored evaluation results.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate() first.")
        return self.evaluation_results