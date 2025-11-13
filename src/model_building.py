import os
import sys
from abc import ABC, abstractmethod
from typing import Optional
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier,GBTClassifier
from pyspark.ml import PipelineModel
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
from logger import get_logger
from spark_session  import get_or_create_spark_session
logger = get_logger(__name__)

class BaseModelBuilder(ABC):
    """
    Abstract base class for all machine learning model builders.

    Attributes
    ----------
    model_name : str
        Name of the model.
    model : object
        Model object (initialized in build_model).
    model_params : dict
        Dictionary of model parameters for building the model.
    """

    def __init__(self, model_name: str, featuresCol:str,labelCol:Optional[str], spark: Optional[SparkSession] = None,**kwargs):
        """
        Initialize the base model builder.

        Args:
            model_name (str): Name of the model.
            **kwargs: Arbitrary keyword arguments for model parameters.
        """
        self.spark = spark or get_or_create_spark_session()
        self.model_name = model_name
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.model = None
        self.model_params = kwargs

    @abstractmethod
    def build_model(self):
        """
        Build or initialize the model.

        This method should be implemented in child classes to initialize the
        model with the appropriate parameters. Training is handled separately.
        """
        pass

    def save_model(self, save_path: str):
        """
        Save the model to the specified path.

        Args:
            save_path (str): Directory path where the model should be saved.

        Raises:
            ValueError: If no model is initialized.
            Exception: If saving fails for any other reason.
        """
        if self.model is None:
            raise ValueError("No model to save. Build the model first.")
        try:
            self.model.save(save_path)
            logger.info(f"{self.model_name} model successfully saved at {save_path}")
        except Exception as e:
            logger.exception(f"Error saving {self.model_name} model: {e}")
            raise

    def load_model(self, load_path: str):
        """
        Load a saved model from the specified path.

        Args:
            load_path (str): Directory path from which to load the model.

        Raises:
            ValueError: If the path does not exist.
            Exception: If loading fails for any other reason.
        """
        if not os.path.exists(load_path):
            raise ValueError(f"Can't load. Path not found: {load_path}")
        try:
            self.model = PipelineModel.load(load_path)
            logger.info(f"{self.model_name} model successfully loaded from {load_path}")
        except Exception as e:
            logger.exception(f"Error loading {self.model_name} model: {e}")
            raise


class RandomForestModelBuilder(BaseModelBuilder):
    """
    RandomForest model builder using PySpark ML.

    Attributes
    ----------
    model_name : str
        'RandomForest'.
    model : RandomForestClassifier
        PySpark RandomForestClassifier object.
    model_params : dict
        Parameters used to initialize the RandomForestClassifier.
    """

    def __init__(self, **kwargs):
        """
        Initialize the RandomForest model builder with default or custom parameters.

        Default parameters:
            numTrees: 100
            maxDepth: 10
            seed: 42

        Args:
            **kwargs: Optional custom model parameters to override defaults.
        """
        default_params = {
            'numTrees': 100,
            'maxDepth': 10,
            'seed': 42
        }
        default_params.update(kwargs)
        super().__init__('RandomForest', 'features', 'labels', **default_params)

    def build_model(self):
        """
        Initialize the PySpark RandomForestClassifier with the specified parameters.

        Returns:
            RandomForestClassifier: Initialized RandomForestClassifier object.

        Raises:
            Exception: If model initialization fails.
        """
        try:
            logger.info("Initializing RandomForest model...")
            self.model = RandomForestClassifier(**self.model_params)
            logger.info("RandomForest model initialized successfully.")
            return self.model
        except Exception as e:
            logger.exception(f"Error initializing RandomForest model: {e}")
            raise

class SparkGBTModelBuilder(BaseModelBuilder):
    """
    GBT model builder using PySpark ML.

    Attributes
    ----------
    model_name : str
        'GBT'.
    model : SparkGBTClassifier
        PySpark GBT object.
    model_params : dict
        Parameters used to initialize the GBTClassifier.
    """

    def __init__(self, **kwargs):
        """
        Initialize the RandomForest model builder with default or custom parameters.

        Default parameters:
            'maxDepth': 5,
            'maxIter': 100,
            'seed': 42

        Args:
            **kwargs: Optional custom model parameters to override defaults.
        """
        default_params = {
            'maxDepth': 5,
            'maxIter': 100,
            'seed': 42
        }
        default_params.update(kwargs)
        super().__init__('SparkGBTClassifier', 'features', 'labels', **default_params)

    def build_model(self):
        """
        Initialize the PySpark RandomForestClassifier with the specified parameters.

        Returns:
            RandomForestClassifier: Initialized RandomForestClassifier object.

        Raises:
            Exception: If model initialization fails.
        """
        try:
            logger.info("Initializing GBTclassifier model...")
            self.model = GBTClassifier(**self.model_params)
            logger.info("GBTclassifier model initialized successfully.")
            return self.model
        except Exception as e:
            logger.exception(f"Error initializing GBTclassifier model: {e}")
            raise
