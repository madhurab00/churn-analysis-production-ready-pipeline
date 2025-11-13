import sys
import os
import json
from enum import Enum
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from spark_session import get_or_create_spark_session

# Import custom logger
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger

logger = get_logger(__name__)


class FeatureEncodingStrategy(ABC):
    """Abstract base class for all feature encoding strategies."""

    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize with a SparkSession.

        Args:
            spark (Optional[SparkSession]): Spark session instance. If None, a new one is created.
        """
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def encode(self, df: DataFrame) -> DataFrame:
        """
        Encode features in the given DataFrame.

        Args:
            df (DataFrame): PySpark DataFrame to encode.

        Returns:
            DataFrame: Encoded DataFrame.
        """
        pass


class VariableType(str, Enum):
    """Enumeration of variable types."""
    NOMINAL = "nominal"
    ORDINAL = "ordinal"


class NominalEncodingStrategy(FeatureEncodingStrategy):
    """Encodes nominal (categorical, unordered) variables using StringIndexer."""

    def __init__(
        self,
        nominal_columns: List[str],
        one_hot: bool = False,
        spark: Optional[SparkSession] = None,
    ):
        """
        Initialize nominal encoding strategy.

        Args:
            nominal_columns (List[str]): List of column names to encode.
            one_hot (bool): Whether to apply one-hot encoding after indexing.
            spark (Optional[SparkSession]): Spark session instance.
        """
        super().__init__(spark)
        self.nominal_columns = nominal_columns
        self.one_hot = one_hot
        self.encoder_dicts: Dict[str, Dict[str, int]] = {}
        self.indexers: Dict[str, StringIndexer] = {}
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.artifacts_dir = "artifacts/encode"
        os.makedirs(self.artifacts_dir, exist_ok=True)

        logger.info(f"Initialized NominalEncodingStrategy for columns: {nominal_columns}")
        logger.info(f"One-hot encoding: {one_hot}")

    def encode(self, df: DataFrame) -> DataFrame:
        try:
            logger.info(f"\n{'='*60}")
            logger.info("FEATURE ENCODING - NOMINAL VARIABLES")
            logger.info(f"{'='*60}")

            df_encoded = df

            for column in self.nominal_columns:
                if column not in df_encoded.columns:
                    logger.warning(f"Column '{column}' not in DataFrame — skipping.")
                    continue

                logger.info(f"Encoding nominal variable '{column}'")

                indexer = StringIndexer(
                    inputCol=column,
                    outputCol=f"{column}_index",
                    handleInvalid="keep"
                )

                indexer_model = indexer.fit(df_encoded)
                self.indexers[column] = indexer_model

                labels = indexer_model.labels
                mapping = {label: idx for idx, label in enumerate(labels)}
                self.encoder_dicts[column] = mapping

                # Save mapping
                mapping_path = os.path.join(self.artifacts_dir, f"{column}_mapping.json")
                try:
                    with open(mapping_path, "w") as f:
                        json.dump(mapping, f, indent=4)
                    logger.info(f"✓ Saved encoder mapping for '{column}' → {mapping_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save mapping for '{column}': {e}")
                    raise

                df_encoded = indexer_model.transform(df_encoded).drop(column)
                logger.info(f"✓ Column '{column}' encoded with {len(labels)} unique categories")

            return df_encoded

        except Exception as e:
            logger.error(f"❌ Failed to encode nominal features {self.nominal_columns}: {str(e)}")
            raise



    def get_encoder_dicts(self) -> Dict[str, Dict[str, int]]:
        """
        Get encoder mappings for all columns.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary of column-to-category mappings.
        """
        return self.encoder_dicts

    def get_indexers(self) -> Dict[str, StringIndexer]:
        """
        Get fitted StringIndexer models for each nominal column.

        Returns:
            Dict[str, StringIndexer]: Dictionary of StringIndexer models.
        """
        return self.indexers


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    """Encodes ordinal variables using predefined value-order mappings."""

    def __init__(
        self, ordinal_mappings: Dict[str, Dict[str, int]], spark: Optional[SparkSession] = None
    ):
        """
        Initialize ordinal encoding strategy.

        Args:
            ordinal_mappings (Dict[str, Dict[str, int]]): Mapping of column names to value→order dictionaries.
            spark (Optional[SparkSession]): Spark session instance.
        """
        super().__init__(spark)
        self.ordinal_mappings = ordinal_mappings
        self.artifacts_dir = "artifacts/encode"
        os.makedirs(self.artifacts_dir, exist_ok=True)

        logger.info(f"Initialized OrdinalEncodingStrategy for columns: {list(ordinal_mappings.keys())}")

    def encode(self, df: DataFrame) -> DataFrame:
        """
        Apply ordinal encoding to specified columns.

        Args:
            df (DataFrame): Input PySpark DataFrame.

        Returns:
            DataFrame: DataFrame with ordinal-encoded columns.
        """
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info("FEATURE ENCODING - ORDINAL VARIABLES")
            logger.info(f"{'=' * 60}")

            df_encoded = df

            for column, mapping in self.ordinal_mappings.items():
                logger.info(f"Encoding ordinal variable '{column}' with {len(mapping)} categories")

                # Build mapping expression
                mapping_expr = F.create_map([F.lit(x) for kv in mapping.items() for x in kv])
                df_encoded = df_encoded.withColumn(f"{column}_encoded", mapping_expr[F.col(column)]).drop(column)

            return df_encoded

        except Exception as e:
            logger.error(f"Failed to map ordinal features: {str(e)}")
            raise


class OneHotEncodingStrategy(FeatureEncodingStrategy):
    """Encodes categorical variables using one-hot encoding."""

    def __init__(
        self,
        categorical_columns: List[str],
        max_categories: int = 50,
        spark: Optional[SparkSession] = None,
    ):
        """
        Initialize one-hot encoding strategy.

        Args:
            categorical_columns (List[str]): List of categorical columns to encode.
            max_categories (int): Maximum categories to encode per column.
            spark (Optional[SparkSession]): Spark session instance.
        """
        super().__init__(spark)
        self.categorical_columns = categorical_columns
        self.max_categories = max_categories
        logger.info(f"Initialized OneHotEncodingStrategy for columns: {categorical_columns}")

    def encode(self, df: DataFrame) -> DataFrame:
        """
        Apply one-hot encoding to specified columns.

        Args:
            df (DataFrame): Input PySpark DataFrame.

        Returns:
            DataFrame: DataFrame with one-hot encoded columns.
        """
        logger.info(f"\n{'=' * 60}")
        logger.info("ONE-HOT ENCODING (PySpark)")
        logger.info(f"{'=' * 60}")

        df_encoded = df

        for column in self.categorical_columns:
            unique_count = df_encoded.select(column).distinct().count()

            if unique_count > self.max_categories:
                logger.warning(
                    f"Column '{column}' has {unique_count} categories, exceeding max {self.max_categories}"
                )
                continue

            indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{column}_index", outputCol=f"{column}_vec")

            pipeline = Pipeline(stages=[indexer, encoder])
            pipeline_model = pipeline.fit(df_encoded)
            df_encoded = pipeline_model.transform(df_encoded).drop(column, f"{column}_index")

            logger.info(f"✓ One-hot encoded column '{column}' with {unique_count} categories")

        logger.info(f"✓ ONE-HOT ENCODING COMPLETE")
        logger.info(f"{'=' * 60}\n")
        return df_encoded
