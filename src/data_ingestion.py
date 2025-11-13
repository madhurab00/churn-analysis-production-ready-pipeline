import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from spark_session import get_or_create_spark_session
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger
logger = get_logger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestion supporting both pandas and PySpark."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize DataIngestor with a SparkSession.
        
        Args:
            spark: Optional SparkSession. If not provided, will create/get one.
        """
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> DataFrame:
        """
        Ingest data from the specified path.
        
        Args:
            file_path_or_link: Path to the data file
            
        Returns:
            DataFrame (PySpark or pandas depending on implementation)
        """
        pass


class DataIngestorCSV(DataIngestor):
    """CSV data ingestion implementation."""
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        """
        Ingest CSV data using PySpark.
        
        Args:
            file_path_or_link: Path to the CSV file
            **options: Additional options for CSV reading
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - CSV (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting CSV data ingestion from: {file_path_or_link}")
        
        try:
            # Default CSV options
            csv_options = {
                        "header": "true",
                        "inferSchema": "true",
                        "ignoreLeadingWhiteSpace": "true",
                        "ignoreTrailingWhiteSpace": "true",
                        "nullValue": "",
                        "nanValue": "NaN",
                        "escape": '"',
                        "quote": '"'
                        }
            csv_options.update(options)
            df = self.spark.read.options(**csv_options).csv(file_path_or_link)
            row_count = df.count()
            columns = df.columns
            logger.info(f"✓ Successfully loaded CSV data - Shape: ({row_count}, {len(columns)})")
            logger.info(f"✓ Columns: {columns}")

            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load CSV data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorExcel(DataIngestor):
    """Excel data ingestion implementation."""
    
    def ingest(self, file_path_or_link: str, sheet_name: Optional[str] = None, **options) -> DataFrame:
        """
        Ingest Excel data using PySpark.
        Note: This implementation converts Excel to CSV format internally as PySpark
        doesn't have native Excel support. For production use, consider using
        spark-excel library.
        
        Args:
            file_path_or_link: Path to the Excel file
            sheet_name: Name of the sheet to read (optional)
            **options: Additional options
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - EXCEL (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Excel data ingestion from: {file_path_or_link}")
        
        try:
            logger.info("⚠ Note: Using pandas for Excel reading, then converting to PySpark")
            excel_options={
                'header':'true',
                'inferSchema':'true'
            }
            excel_options.update(options)

            pandas_df = pd.read_excel(file_path_or_link)
            df = self.spark.options(**excel_options).createDataFrame(pandas_df)

            row_count = df.count()
            columns = df.columns
            logger.info(f"✓ Successfully loaded Excel data - Shape: ({row_count}, {len(columns)})")
            logger.info(f"✓ Columns: {columns}")
            logger.info(f"{'='*60}\n")

            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load Excel data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorParquet(DataIngestor):
    """PySpark Parquet data ingestion implementation"""
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        """
        Ingest Parquet data using PySpark.
        Note: Parquet is a columnar format optimized for big data processing.
        
        Args:
            file_path_or_link: Path to the Parquet file or directory
            **options: Additional options for Parquet reading
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - PARQUET (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Parquet data ingestion from: {file_path_or_link}")
        
        try:
            # Read Parquet file(s)
            parquet_options={
                "MergeSchema":"false",
                "RecursiveFileLookup":"true"
            }
            parquet_options.update(options)
            df = self.spark.read.options(**parquet_options).parquet(file_path_or_link)
            row_count = df.count()
            columns = df.columns
            
            logger.info(f"✓ Successfully loaded Parquet data - Shape: ({row_count}, {len(columns)})")
            logger.info(f"✓ Columns: {columns}")
            logger.info(f"✓ Partitions: {df.rdd.getNumPartitions()}")
            logger.info(f"✓ Schema: {df.schema.simpleString()}")
            logger.info(f"{'='*60}\n")


            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load Parquet data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorFactory:
    """Factory class to create appropriate data ingestor based on file type."""
    
    @staticmethod
    def get_ingestor(file_path: str, spark: Optional[SparkSession] = None) -> DataIngestor:
        """
        Get appropriate data ingestor based on file extension.
        
        Args:
            file_path: Path to the data file
            spark: Optional SparkSession
            
        Returns:
            DataIngestor: Appropriate ingestor instance
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return DataIngestorCSV(spark)
        elif file_extension in ['.xlsx', '.xls']:
            return DataIngestorExcel(spark)
        elif file_extension == '.parquet':
            return DataIngestorParquet(spark)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")