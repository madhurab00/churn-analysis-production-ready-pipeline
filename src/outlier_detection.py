import os
from abc import ABC, abstractmethod
import sys
from typing import List, Optional, Dict, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from spark_session import get_or_create_spark_session
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger
logger=get_logger(__name__)


class OutlierDetectionStrategy(ABC):
    """Abstract base class for outlier detection strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def detect_outliers(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """
        Detect outliers in specified columns.
        
        Args:
            df: DataFrame (PySpark or pandas)
            columns: List of column names to check for outliers
            
        Returns:
            DataFrame with additional boolean columns indicating outliers
        """
        pass
    
    @abstractmethod
    def get_outlier_bounds(self, df: DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Get outlier bounds for specified columns.
        
        Args:
            df: DataFrame (PySpark or pandas)
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to (lower_bound, upper_bound) tuples
        """
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    """IQR-based outlier detection strategy."""
    
    def __init__(self, threshold: float = 1.5, spark: Optional[SparkSession] = None):
        """
        Initialize IQR outlier detection.
        
        Args:
            threshold: IQR multiplier for outlier bounds (default: 1.5)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.threshold = threshold
        logger.info(f"Initialized IQROutlierDetection with threshold: {threshold}")
    
    def _get_outlier_bounds(self, df: DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate outlier bounds using IQR method.
        
        Args:
            df: PySpark DataFrame
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to (lower_bound, upper_bound) tuples
        """
        try:
            bounds = {}
            
            for col in columns:
                quantiles = df.approxQuantile(self.relevant_column, [0.25, 0.75], 0.01)
                Q1, Q3 = quantiles[0], quantiles[1]
                IQR = Q3 - Q1

                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                
                bounds[col] = (lower_bound, upper_bound)
                logger.info(f'Upper bound and lower bound for the column:{col} : {upper_bound},{lower_bound}')
            
            return bounds
        
        except Exception as e:
            logger.info(f'Failed to get Outlier bounds')
            raise
        
        
    def detect_outliers(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df: PySpark DataFrame
            columns: List of column names to check for outliers
            
        Returns:
            DataFrame with additional boolean columns '{col}_outlier' indicating outliers
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"OUTLIER DETECTION - IQR METHOD (PySpark)")
            logger.info(f"{'='*60}")
            logger.info(f"Starting IQR outlier detection for columns: {columns}")
            
            # Get outlier bounds
            bounds = self._get_outlier_bounds(df, columns)
            
            # Add outlier indicator columns
            result_df = df
            total_outliers = 0
            
            for col in columns:
                logger.info(f"\n--- Processing column: {col} ---")
                lower_bound, upper_bound = bounds[col]
                
                # Create outlier indicator column
                outlier_col = f"{col}_outlier"
                result_df = result_df.withColumn(
                                                outlier_col,
                                                (F.col(col) < lower_bound) | (F.col(col) > upper_bound)
                                                )

                outlier_count = result_df.filter(F.col(outlier_col)).count()
                total_outliers +=outlier_count

            logger.info(f"\n{'='*60}")
            logger.info(f'✓ OUTLIER DETECTION COMPLETE - Total outlier instances: {total_outliers}')
            logger.info(f"{'='*60}\n")
            
            return result_df
        except Exception as e:
            logger.info(f'Failed to detect outliers in columns:{columns}')

class OutlierDetector:
    """Main outlier detector class that uses different strategies."""
    
    def __init__(self, strategy: OutlierDetectionStrategy):
        """
        Initialize outlier detector with a specific strategy.
        
        Args:
            strategy: OutlierDetectionStrategy instance
        """
        self._strategy = strategy
        logger.info(f"OutlierDetector initialized with strategy: {strategy.__class__.__name__}")
    
    def detect_outliers(self, df: DataFrame, selected_columns: List[str]) -> DataFrame:
        """
        Detect outliers in selected columns.
        
        Args:
            df: DataFrame (PySpark or pandas)
            selected_columns: List of column names to check
            
        Returns:
            DataFrame with outlier indicator columns
        """
        try:
            logger.info(f"Detecting outliers in {len(selected_columns)} columns")
            return self._strategy.detect_outliers(df, selected_columns)
        except Exception as e:
            logger.info(f'Failed to detect outlier :{str(e)} ')
            raise
    
    def handle_outliers(self, df: DataFrame, selected_columns: List[str], 
                       method: str = 'remove', min_outliers: int = 2) -> DataFrame:
        """
        Handle outliers using specified method.
        
        Args:
            df: DataFrame (PySpark or pandas)
            selected_columns: List of column names to check
            method: Method to handle outliers ('remove' or 'cap')
            min_outliers: Minimum number of outlier columns to remove a row
            
        Returns:
            DataFrame with outliers handled
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"OUTLIER HANDLING - {method.upper()} (PySpark)")
            logger.info(f"{'='*60}")
            logger.info(f"Handling outliers using method: {method}")
            
            initial_rows = df.count()
            
            if method == 'remove':
                # Add outlier indicator columns
                df_with_outliers = self.detect_outliers(df, selected_columns)
                
                # Count outliers per row
                outlier_columns = [f"{col}_outlier" for col in selected_columns]
                outlier_count_expr = sum(F.col(col).cast('int') for col in outlier_columns)
                df_with_count = df_with_outliers.withColumn('outlier_count', outlier_count_expr)
                clean_df = df_with_count.filter(F.col("outlier_count") < min_outliers)
                clean_df = clean_df.drop("outlier_count")
                rows_removed = initial_rows - clean_df.count()
                
            elif method == 'cap':
                bounds = self._strategy.get_outlier_bounds(df, selected_columns)
                clean_df = df

                for col in selected_columns:
                    lb, ub = bounds[col]

                    clean_df = clean_df.withColumn(
                                                col,
                                                F.when(F.col(col) < lb, lb)
                                                .when(F.col(col) > ub, ub)
                                                .otherwise(F.col(col))
                                                )
                
            else:
                raise ValueError(f"Unknown outlier handling method: {method}")
            
            logger.info(f"{'='*60}\n")
            return clean_df
        except Exception as e:
            logger.info(f'Failed to handle outliers in columns {selected_columns} :{str(e)} ')
            raise