import os
from abc import ABC, abstractmethod
import sys
from typing import Dict, List, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Bucketizer
from spark_session import get_or_create_spark_session
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger
logger = get_logger(__name__)

class FeatureBinningStrategy(ABC):
    """Abstract base class for feature binning strategies."""

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Bin a feature column.
        
        Args:
            df: PySpark DataFrame
            column: Column name to bin
            
        Returns:
            DataFrame with binned feature
        """
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    """Custom binning strategy with named bins."""
    def __init__(self, bin_definitions: Dict[str, List[float]], spark: Optional[SparkSession] = None):
        """
        Initialize custom binning strategy.
        
        Args:
            bin_definitions: Dictionary mapping bin names to [min, max] ranges
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.bin_definitions = bin_definitions
        logger.info(f"CustomBinningStratergy initialized with bins: {list(bin_definitions)}")
    
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Apply custom binning to a feature column.
        
        Args:
            df: PySpark DataFrame
            column: Column name to bin
            
        Returns:
            DataFrame with original column replaced by binned version
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info("FEATURE BINNING")
            logger.info(f"{'='*60}")
            stats = df.select(
            F.count(F.col(column)).alias('count'),
            F.countDistinct(F.col(column)).alias('unique'),
            F.min(F.col(column)).alias('min'),
            F.max(F.col(column)).alias('max')
            ).collect()[0]
        
            logger.info(f"  Unique values: {stats['unique']}, Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
            bin_column = f'{column}Bins'
            case_expr = None
            for bin_label, bin_range in self.bin_definitions.items():
                if case_expr is None:
                    if len(bin_range) == 2:
                        case_expr = F.when(
                                                (F.col(column) >= bin_range[0]) & (F.col(column) <= bin_range[1]), 
                                                bin_label
                                                )
                    elif len(bin_range) == 1:
                        case_expr = F.when(
                                                (F.col(column) >= bin_range[0]), 
                                                bin_label
                                                )
                else:
                    if len(bin_range) == 2:
                        case_expr = case_expr.when(
                                                (F.col(column) >= bin_range[0]) & (F.col(column) <= bin_range[1]), 
                                                bin_label
                                                )
                    elif len(bin_range) == 1:
                        case_expr = case_expr.when(
                                                (F.col(column) >= bin_range[0]), 
                                                bin_label
                                                )


            df_binned = df.withColumn(bin_column, case_expr)
            df_binned = df_binned.drop(column)
            logger.info(f"✓ Added '{bin_column}' column (keeping original '{column}' for model compatibility)")
            logger.info(f"{'='*60}\n")

            return df_binned
        except Exception as e:
            logger.info(f'Failed to bin column {column}:{str(e)}')
            raise

class BucketizerBinningStrategy(FeatureBinningStrategy):
    """Binning strategy using PySpark's Bucketizer."""
    
    def __init__(self, splits: List[float], labels: Optional[List[str]] = None, 
                 handle_invalid: str = "keep", spark: Optional[SparkSession] = None):
        """
        Initialize Bucketizer binning strategy.
        
        Args:
            splits: List of split points for binning (must be monotonically increasing)
            labels: Optional list of bin labels (length should be len(splits) - 1)
            handle_invalid: How to handle values outside splits ("keep", "skip", "error")
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.splits = splits
        self.labels = labels
        self.handle_invalid = handle_invalid
        logger.info(f"BucketizerBinningStrategy initialized with {len(splits)-1} bins")

    def bin_feature(self, df: DataFrame, column:str) ->DataFrame:
        """
        Apply Bucketizer binning to a feature column.
        
        Args:
            df: PySpark DataFrame
            column: Column name to bin
            
        Returns:
            DataFrame with binned feature
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"BUCKETIZER BINNING - {column.upper()}")
        logger.info(f"{'='*60}")

        bin_column = f"{column}Bins"
        
        bucketizer = Bucketizer(
            splits=self.splits,
            inputCol=column,
            outputCol=bin_column,
            handleInvalid=self.handle_invalid
        )
        
        df_binned = bucketizer.transform(df)
        logger.info(f"✓ Binning complete for column '{column}' (original column preserved)")
        logger.info(f"{'='*60}\n")

        return df_binned