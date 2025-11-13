import os
import sys
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.functions import vector_to_array
from spark_session import get_or_create_spark_session

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger

logger = get_logger(__name__)

class FeatureScalingStrategy(ABC):
    """Abstract base class for feature scaling strategies."""

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession and artifacts directory."""
        self.spark = spark or get_or_create_spark_session()
        self.fitted_models: Dict[str, Pipeline] = {}
        self.artifacts_dir = "artifacts/scalers"
        os.makedirs(self.artifacts_dir, exist_ok=True)

    @abstractmethod
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """Scale specified columns in the DataFrame."""
        pass

    def save_scaler(self, column_name: str):
        """Save a fitted scaler pipeline for a given column."""
        if column_name not in self.fitted_models:
            raise ValueError(f"No fitted scaler found for column '{column_name}'")
        save_path = os.path.join(self.artifacts_dir, f"{column_name}_scaler")
        self.fitted_models[column_name].write().overwrite().save(save_path)
        logger.info(f"✓ Saved scaler for column '{column_name}' at {save_path}")

    def load_scaler(self, column_name: str) -> PipelineModel:
        """Load a saved scaler pipeline for a given column."""
        path = os.path.join(self.artifacts_dir, f"{column_name}_scaler")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler for column '{column_name}' not found at {path}")
        pipeline_model = PipelineModel.load(path)
        logger.info(f"✓ Loaded scaler for column '{column_name}' from {path}")
        return pipeline_model


class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """Min-Max scaling strategy to scale features to [0, 1] range."""

    def __init__(self, output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.output_col_suffix = output_col_suffix
        logger.info("MinMaxScalingStrategy initialized (PySpark)")

    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """Apply Min-Max scaling to specified columns."""
        try:
            logger.info(f"\n{'='*60}")
            logger.info("FEATURE SCALING - MINMAX SCALING")
            logger.info(f"{'='*60}")
            df_scaled = df

            for col in columns_to_scale:
                vector_col = f"{col}_vec"
                scaled_vector_col = f"{col}{self.output_col_suffix}_vec"

                # Create pipeline: assembler + scaler
                assembler = VectorAssembler(inputCols=[col], outputCol=vector_col)
                scaler = MinMaxScaler(inputCol=vector_col, outputCol=scaled_vector_col)
                pipeline = Pipeline(stages=[assembler, scaler])

                # Fit and transform
                pipeline_model = pipeline.fit(df_scaled)
                df_scaled = pipeline_model.transform(df_scaled)

                # Replace original column with scaled values
                df_scaled = df_scaled.withColumn(
                    col, vector_to_array(F.col(scaled_vector_col)).getItem(0)
                ).drop(vector_col, scaled_vector_col)

                # Save fitted scaler
                self.fitted_models[col] = pipeline_model
                self.save_scaler(col)

            return df_scaled

        except Exception as e:
            logger.error(f"Failed to scale columns {columns_to_scale} using MinMax scaling: {e}")
            raise


class StandardScalingStrategy(FeatureScalingStrategy):
    """Standard scaling strategy to scale features to zero mean and unit variance."""

    def __init__(self, with_mean: bool = True, with_std: bool = True,
                 output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.with_mean = with_mean
        self.with_std = with_std
        self.output_col_suffix = output_col_suffix
        logger.info(f"StandardScalingStrategy initialized (PySpark) - with_mean={with_mean}, with_std={with_std}")

    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """Apply Standard scaling to specified columns."""
        try:
            logger.info(f"\n{'='*60}")
            logger.info("FEATURE SCALING - STANDARD SCALING")
            logger.info(f"{'='*60}")
            df_scaled = df

            for col in columns_to_scale:
                vector_col = f"{col}_vec"
                scaled_vector_col = f"{col}{self.output_col_suffix}_vec"

                # Create pipeline: assembler + scaler
                assembler = VectorAssembler(inputCols=[col], outputCol=vector_col)
                scaler = StandardScaler(
                    inputCol=vector_col,
                    outputCol=scaled_vector_col,
                    withMean=self.with_mean,
                    withStd=self.with_std
                )
                pipeline = Pipeline(stages=[assembler, scaler])

                # Fit and transform
                pipeline_model = pipeline.fit(df_scaled)
                df_scaled = pipeline_model.transform(df_scaled)

                # Replace original column with scaled values
                df_scaled = df_scaled.withColumn(
                    col, vector_to_array(F.col(scaled_vector_col)).getItem(0)
                ).drop(vector_col, scaled_vector_col)

                # Save fitted scaler
                self.fitted_models[col] = pipeline_model
                self.save_scaler(col)

            return df_scaled

        except Exception as e:
            logger.error(f"Failed to scale columns {columns_to_scale} using Standard scaling: {e}")
            raise
