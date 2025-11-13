import os
import sys
import logging
import json
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from spark_session import create_spark_session, stop_spark_session
from spark_utils import save_dataframe, spark_to_pandas, get_dataframe_info
from data_ingestion import DataIngestorCSV
from handle_missing_values import FillMissingValuesStrategy
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleTrainTestSplitStrategy

from config import (
    get_data_paths, get_columns, get_missing_values_config, get_outlier_config,
    get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config
)
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
from logger import get_logger

logger = get_logger(__name__)


def save_processed_data(
    train_df: DataFrame, 
    test_df: DataFrame, 
    output_format: str = "both"
) -> Dict[str, str]:
    """Save processed data in specified format(s)."""
    os.makedirs('artifacts/data', exist_ok=True)
    paths = {}

    if output_format in ["csv", "both"]:
        logger.info("Saving data in CSV format...")
        train_pd = spark_to_pandas(train_df)
        test_pd = spark_to_pandas(test_df)

        paths['train_csv'] = 'artifacts/data/train.csv'
        paths['test_csv'] = 'artifacts/data/test.csv'

        train_pd.to_csv(paths['train_csv'], index=False)
        test_pd.to_csv(paths['test_csv'], index=False)
        logger.info("✓ CSV files saved")

    if output_format in ["parquet", "both"]:
        logger.info("Saving data in Parquet format...")
        paths['train_parquet'] = 'artifacts/data/train.parquet'
        paths['test_parquet'] = 'artifacts/data/test.parquet'

        save_dataframe(train_df, paths['train_parquet'], format='parquet')
        save_dataframe(test_df, paths['test_parquet'], format='parquet')
        logger.info("✓ Parquet files saved")

    return paths


def data_pipeline(
    data_path: str = 'data/raw/churndataset.csv',
    target_column: str = 'Churn',
    test_size: float = 0.2,
    force_rebuild: bool = False,
    output_format: str = "both"
) -> Dict[str, np.ndarray]:
    """Execute comprehensive data processing pipeline with PySpark and MLflow tracking."""
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING PYSPARK DATA PIPELINE")
    logger.info(f"{'='*80}")

    if not os.path.exists(data_path):
        logger.error(f"✗ Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not 0 < test_size < 1:
        logger.error(f"✗ Invalid test_size: {test_size}")
        raise ValueError(f"Invalid test_size: {test_size}")

    spark = create_spark_session("ChurnPredictionDataPipeline")

    try:
        # Load configurations
        data_paths = get_data_paths()
        columns = get_columns()
        missing_value_config = get_missing_values_config()
        outlier_config = get_outlier_config()
        binning_config = get_binning_config()
        encoding_config = get_encoding_config()
        scaling_config = get_scaling_config()
        splitting_config = get_splitting_config()

        # Initialize MLflow tracking
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags('data_pipeline_pyspark', {
            'data_source': data_path,
            'force_rebuild': str(force_rebuild),
            'target_column': target_column,
            'output_format': output_format,
            'processing_engine': 'pyspark'
        })
        run = mlflow_tracker.start_run(run_name='data_pipeline_pyspark', tags=run_tags)

        # Create artifacts directory
        run_artifacts_dir = os.path.join('artifacts', 'mlflow_run_artifacts', run.info.run_id)
        os.makedirs(run_artifacts_dir, exist_ok=True)

        # Check for existing artifacts
        train_path = os.path.join('artifacts', 'data', 'train.csv')
        test_path = os.path.join('artifacts', 'data', 'test.csv')
        artifacts_exist = all(os.path.exists(p) for p in [train_path, test_path])

        if artifacts_exist and not force_rebuild:
            logger.info("✓ Loading existing processed data artifacts")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            mlflow_tracker.log_data_pipeline_metrics({
                'total_samples': len(train_df) + len(test_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'processing_engine': 'existing_artifacts'
            })
            mlflow_tracker.end_run()

            logger.info("✓ Data pipeline completed using existing artifacts")
            return {
                'train': train_df.values,
                'test': test_df.values
            }

        # Process data from scratch
        logger.info("Processing data from scratch with PySpark...")

        # Data ingestion
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA INGESTION STEP")
        logger.info(f"{'='*80}")
        ingestor = DataIngestorCSV(spark)
        df = ingestor.ingest(data_path)
        logger.info(f"✓ Raw data loaded: {get_dataframe_info(df)}")
        mlflow_tracker.log_stage_metrics(df, stage='raw')

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Handle missing values
        logger.info(f"\n{'='*80}")
        logger.info(f"HANDLING MISSING VALUES STEP")
        logger.info(f"{'='*80}")
        initial_count = df.count()
        fill_handler = FillMissingValuesStrategy(method='mean', relevant_column='TotalCharges', spark=spark)
        df = fill_handler.handle(df)
        rows_removed = initial_count - df.count()
        mlflow_tracker.log_stage_metrics(df, stage='missing_handled', additional_metrics={'rows_removed': rows_removed})
        logger.info(f"✓ Missing values handled: {initial_count} → {df.count()}")

        # Feature binning
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE BINNING STEP")
        logger.info(f"{'='*80}")
        binning = CustomBinningStrategy(binning_config['tenure_bins'], spark=spark)
        df = binning.bin_feature(df, 'tenure')

        # Log binning distribution
        if 'tenureBins' in df.columns:
            bin_dist = df.groupBy('tenureBins').count().collect()
            bin_metrics = {f'tenure_bin_{row["tenureBins"]}': row['count'] for row in bin_dist}
            mlflow.log_metrics(bin_metrics)
        logger.info("✓ Feature binning completed")
        

        # Feature encoding
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE ENCODING STEP")
        logger.info(f"{'='*80}")
        nominal_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'], spark=spark)
        ordinal_strategy = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'], spark=spark)
        df = nominal_strategy.encode(df)
        df = ordinal_strategy.encode(df)
        mlflow_tracker.log_stage_metrics(df, stage='encoded')
        logger.info("✓ Feature encoding completed")

        # Feature scaling
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE SCALING STEP")
        logger.info(f"{'='*80}")
        minmax_strategy = MinMaxScalingStrategy(spark=spark)
        df = minmax_strategy.scale(df, scaling_config['columns_to_scale'])
        logger.info("✓ Feature scaling completed")

        # Drop unnecessary columns
        drop_columns = columns['drop_columns']
        existing_drop_columns = [col for col in drop_columns if col in df.columns]
        if existing_drop_columns:
            df = df.drop(*existing_drop_columns)
            logger.info(f"✓ Dropped columns: {existing_drop_columns}")

        # After all encoding, scaling, and dropping unnecessary columns:
        all_cols = df.columns
        if "Churn_index" in all_cols:
            # Put Churn_index at the end
            other_cols = [c for c in all_cols if c != "Churn_index"]
            df = df.select(other_cols + ["Churn_index"])

        # Data splitting
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA SPLITTING STEP")
        logger.info(f"{'='*80}")
        splitting_strategy = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'], spark=spark)
        train_df, test_df = splitting_strategy.split_data(df)

        # Save processed data
        output_paths = save_processed_data(train_df, test_df, output_format)
        mlflow_tracker.log_stage_metrics(train_df, stage='final_train')
        mlflow_tracker.log_stage_metrics(test_df, stage='final_test')
        logger.info("✓ Data splitting completed")

        # Save preprocessing metadata
        if hasattr(minmax_strategy, 'scaler_models'):
            model_path = os.path.join('artifacts', 'encode', 'fitted_preprocessing_model')
            os.makedirs(model_path, exist_ok=True)
            preprocessing_metadata = {
                'scaling_columns': scaling_config['columns_to_scale'],
                'encoding_columns': encoding_config['nominal_columns'],
                'ordinal_mappings': encoding_config['ordinal_mappings'],
                'binning_config': binning_config,
                'spark_version': spark.version
            }
            with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                json.dump(preprocessing_metadata, f, indent=2)
            logger.info(f"✓ Saved preprocessing metadata to {model_path}")

        # Log comprehensive pipeline metrics
        comprehensive_metrics = {
            'total_samples': train_df.count() + test_df.count(),
            'train_samples': train_df.count(),
            'test_samples': test_df.count(),
            'final_features': len(train_df.columns),
            'processing_engine': 'pyspark',
            'output_format': output_format
        }
        mlflow.log_params({
            'final_feature_names': train_df.columns,
            'preprocessing_steps': ['missing_values', 'outlier_detection', 'feature_binning',
                                  'feature_encoding', 'feature_scaling'],
            'data_pipeline_version': '3.0_pyspark'
        })

        # Log artifacts
        for path_key, path_value in output_paths.items():
            if os.path.exists(path_value):
                mlflow.log_artifact(path_value, "processed_datasets")

        mlflow_tracker.end_run()

        # Convert to numpy arrays
        train_np = spark_to_pandas(train_df).values
        test_np = spark_to_pandas(test_df).values

        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL DATASET SHAPES")
        logger.info(f"{'='*80}")
        logger.info(f"✓ Final dataset shapes:")
        logger.info(f"  • train shape: {train_np.shape}")
        logger.info(f"  • test shape:  {test_np.shape}")
        logger.info(f"  • Total samples: {train_np.shape[0] + test_np.shape[0]}")
        logger.info(f"  • Train/Test ratio: {train_np.shape[0]/(train_np.shape[0]+test_np.shape[0]):.1%} / {test_np.shape[0]/(train_np.shape[0]+test_np.shape[0]):.1%}")
        logger.info(f"\n{'='*80}")
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY")

        return {'train': train_np, 'test': test_np}

    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        if 'mlflow_tracker' in locals():
            mlflow_tracker.end_run()
        raise
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    processed_data = data_pipeline(output_format="both")
    logger.info(f"Pipeline completed. Train samples: {processed_data['train'].shape[0]}")