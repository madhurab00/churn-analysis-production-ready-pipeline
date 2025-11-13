import os
import sys
from typing import Optional, List, Union
from abc import ABC, abstractmethod
import pandas as pd  
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer
from spark_session import get_or_create_spark_session
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger

logger=get_logger(__name__)

class MissingValueHandlingStrategy(ABC):
    """Abstract base class for missing value handling strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def handle(self, df: DataFrame) -> DataFrame:
        """Handle missing values in the DataFrame."""
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Strategy to drop rows with missing values in critical columns."""
    
    def __init__(self, critical_columns: Optional[List[str]] = None, spark: Optional[SparkSession] = None):
        """
        Initialize the drop strategy.
        
        Args:
            critical_columns: List of column names where nulls are not allowed
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.critical_columns = critical_columns or []
        logger.info(f"Initialized DropMissingValuesStrategy for columns: {self.critical_columns}")

    def handle(self, df: DataFrame) -> DataFrame:
        """
        Drop rows with missing values in critical columns.
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            DataFrame with rows dropped
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info("DROPPING MISSING VALUES")
            logger.info(f"{'='*60}")
            initial_count = df.count()
            
            if self.critical_columns:
                df_cleaned = df.dropna(subset=self.critical_columns)

            else:
                df_cleaned = df.dropna()
            final_count = df_cleaned.count()
            n_dropped = initial_count - final_count

            logger.info(f"✓ Dropped {n_dropped} rows with missing values")
            logger.info(f"  • Initial rows: {initial_count}")
            logger.info(f"  • Final rows: {final_count}")

            return df_cleaned
        
        except Exception as e:
            logger.info(f'Failed to drop missing vlaues on {self.critical_columns}:{str(e)}')
            raise

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to fill missing values using various methods.
    Supports mean/median/mode filling and custom imputers.
    """
    
    def __init__(
        self, 
        method: str = 'mean', 
        fill_value: Optional[Union[str, float, int]] = None, 
        relevant_column: Optional[str] = None, 
        is_custom_imputer: bool = False,
        custom_imputer: Optional[object] = None,
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize the fill strategy.
        
        Args:
            method: Method to use ('mean', 'median', 'mode', 'constant')
            fill_value: Value to use for constant filling
            relevant_column: Column to fill (if None, fills all numeric columns)
            is_custom_imputer: Whether to use a custom imputer
            custom_imputer: Custom imputer object (must have impute method)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer

    def handle(self, df: DataFrame) -> DataFrame:
        """
        Fill missing values based on the configured strategy.
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            DataFrame with filled values
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info("FILLING MISSING VALUES")
            logger.info(f"{'='*60}")
            if self.is_custom_imputer and self.custom_imputer:
                logger.info(f'✓ Filled missing values using {self.custom_imputer}')
                return self.custom_imputer.impute(df)
            
            if self.relevant_column:
                # Fill specific column
                if self.method == 'mean':
                    mean_value = df.select(F.mean(F.col(self.relevant_column))).collect()[0][0]
                    df_filled = df.fillna({self.relevant_column: mean_value})
                    logger.info(f'✓ Filled missing values in {self.relevant_column} with mean: {mean_value:.2f}')
                    
                elif self.method == 'median':
                    median_value = df.approxQuantile(self.relevant_column, [0.5], 0.01)[0]
                    df_filled = df.fillna({self.relevant_column: median_value})
                    logger.info(f'✓ Filled missing values in {self.relevant_column} with median: {median_value:.2f}')      
                    
                elif self.method == 'mode':
                    mode_value = df.groupBy(self.relevant_column).count().orderBy(F.desc('count')).first()[0]
                    df_filled = df.fillna({self.relevant_column: mode_value})
                    logger.info(f'✓ Filled missing values in {self.relevant_column} with mean: {mode_value:.2f}')
                    
                elif self.method == 'constant' and self.fill_value is not None:
                    df_filled = df.fillna({self.relevant_column: self.fill_value})
                    logger.info(f'✓ Filled missing values in {self.relevant_column} with constant: {self.fill_value}')
                    
                else:
                    raise ValueError(f"Invalid method '{self.method}' or missing fill_value")
                    
            else:
                # Fill all columns based on method
                if self.method == 'constant' and self.fill_value is not None:
                    df_filled = df.fillna(self.fill_value)
                    logger.info(f'✓ Filled all missing values with constant: {self.fill_value}')
                else:
                    # Use Spark ML Imputer for mean/median on all numeric columns
                    numeric_cols = [field.name for field in df.schema.fields 
                                if field.dataType.typeName() in ['integer', 'long', 'float', 'double']]
                    
                    if numeric_cols:
                        imputer = Imputer(
                            inputCols=numeric_cols,
                            outputCols=[f"{col}_imputed" for col in numeric_cols],
                            strategy=self.method if self.method in ['mean', 'median'] else 'mean'
                        )
                        
                        model = imputer.fit(df)
                        df_imputed = model.transform(df)
                        
                        # Replace original columns with imputed ones
                        for col in numeric_cols:
                            df_imputed = df_imputed.withColumn(col, F.col(f"{col}_imputed")) \
                                .drop(f"{col}_imputed")
                        
                        df_filled = df_imputed
                        logger.info(f'✓ Filled missing values in numeric columns using {self.method}')
                    else:
                        df_filled = df
                        logger.warning('No numeric columns found for imputation')
            
            return df_filled
        
        except Exception as e:
            logger.info(f'Failed to fill missing vlaues on {self.relevant_column}:{str(e)}')
            raise