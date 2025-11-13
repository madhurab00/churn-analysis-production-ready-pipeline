import os
import sys
import json
import pandas as pd
from typing import Any, Dict, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

# Add utils to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from config import (
    get_binning_config,
    get_columns,
    get_encoding_config,
    get_scaling_config,
)
from spark_session import get_or_create_spark_session
from logger import get_logger

logger = get_logger(__name__)


class ModelInference:
    """Perform model inference using Spark ML pipeline and saved encoders."""

    def __init__(self, model_path: str, artifacts_dir: str = "artifacts/encode", spark: Optional[SparkSession] = None):
        self.model_path = model_path
        self.artifacts_dir = artifacts_dir
        self.spark = spark or get_or_create_spark_session()
        self.model: Optional[PipelineModel] = None
        self.nominal_mappings: Dict[str, Dict[str, int]] = {}
        self._load_model()
        self._load_nominal_mappings()

    def _load_model(self) -> None:
        try:
            self.model = PipelineModel.load(self.model_path)
            logger.info(f"✅ Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _load_nominal_mappings(self) -> None:
        try:
            mapping_files = [f for f in os.listdir(self.artifacts_dir) if f.endswith("_mapping.json")]
            if not mapping_files:
                raise FileNotFoundError("No nominal encoder mappings found in artifacts directory.")

            for file_name in mapping_files:
                column_name = file_name.replace("_mapping.json", "")
                with open(os.path.join(self.artifacts_dir, file_name), "r") as f:
                    self.nominal_mappings[column_name] = json.load(f)
            logger.info(f"✅ Loaded nominal mappings for columns: {list(self.nominal_mappings.keys())}")

        except Exception as e:
            logger.error(f"❌ Failed to load nominal mappings: {e}")
            raise

    def _encode_nominal_features(self, df: DataFrame) -> DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("APPLYING SAVED NOMINAL ENCODERS")
        logger.info(f"{'='*60}")

        df_encoded = df
        for column, mapping in self.nominal_mappings.items():
            if column not in df_encoded.columns:
                logger.warning(f"Column '{column}' not in input DataFrame — skipping.")
                continue

            mapping_expr = F.when(F.col(column).isNull(), None)
            for label, code in mapping.items():
                mapping_expr = mapping_expr.when(F.col(column) == label, F.lit(code))
            mapping_expr = mapping_expr.otherwise(F.lit(-1))

            df_encoded = df_encoded.withColumn(f"{column}_index", mapping_expr).drop(column)
            logger.info(f"✓ Applied saved encoding for '{column}' ({len(mapping)} categories)")

        return df_encoded

    def _preprocess_data(self, data: Dict[str, Any]) -> DataFrame:
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING INPUT DATA")
        logger.info("=" * 60)

        if not isinstance(data, dict) or not data:
            raise ValueError("Input data must be a non-empty dictionary.")

        try:
            pdf = pd.DataFrame([data])
            sdf = self.spark.createDataFrame(pdf)

            expected_columns = get_columns()["feature_columns"]
            missing = set(expected_columns) - set(sdf.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Binning
            bin_handler = CustomBinningStrategy(get_binning_config()["tenure_bins"], spark=self.spark)
            sdf = bin_handler.bin_feature(sdf, "tenure")

            # Ordinal encoding
            ordinal_encoder = OrdinalEncodingStrategy(get_encoding_config()["ordinal_mappings"], spark=self.spark)
            sdf = ordinal_encoder.encode(sdf)

            # Nominal encoding
            sdf = self._encode_nominal_features(sdf)

            # Scaling
            scaling_conf = get_scaling_config()
            scaler = MinMaxScalingStrategy(spark=self.spark)
            for col in scaling_conf["columns_to_scale"]:
                pipeline_model = scaler.load_scaler(col)
                vector_col = f"{col}_vec"
                scaled_vector_col = f"{col}_scaled_vec"

                assembler = VectorAssembler(inputCols=[col], outputCol=vector_col)
                sdf = assembler.transform(sdf)
                sdf = pipeline_model.stages[1].transform(sdf)
                sdf = sdf.withColumn(col, vector_to_array(F.col(scaled_vector_col)).getItem(0))
                sdf = sdf.drop(vector_col, scaled_vector_col)

            # Drop unnecessary columns
            drop_columns = get_columns().get("drop_columns", [])
            existing_drops = [col for col in drop_columns if col in sdf.columns]
            if existing_drops:
                sdf = sdf.drop(*existing_drops)

            # Fill missing columns required by the model
            model_input_cols = self.model.stages[0].getInputCols()
            for col in model_input_cols:
                if col not in sdf.columns:
                    sdf = sdf.withColumn(col, F.lit(0))

            logger.info("✅ Preprocessing completed successfully.")
            return sdf

        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            raise

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sdf = self._preprocess_data(data)

            # Transform directly with model (no extra assembler)
            pred_df = self.model.transform(sdf)
            result_row = pred_df.select("prediction", "probability").collect()[0]

            prediction = int(result_row["prediction"])
            prob = float(result_row["probability"][1]) if hasattr(result_row["probability"], "__getitem__") else 0.0
            label = "Churn" if prediction == 1 else "Retain"
            confidence = round(prob * 100, 2)

            result = {
                "prediction_label": label,
                "confidence": f"{confidence} %",
                "prediction": prediction,
                "probability": prob,
            }

            logger.info(f"✅ Inference result: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            raise
