import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.pipelines.features import FeaturePipeline
from src.pipelines.model import XGBoostModel


class Trainer:
    """
    Orchestrates the training and evaluation process.
    """

    def __init__(self, model: XGBoostModel, features: FeaturePipeline):
        """
        Initializes the Trainer.

        Args:
            model: The machine learning model to be trained.
            features: The feature engineering pipeline to process the data.
        """
        self.model = model
        self.features = features
        self._is_trained = False
        self.metrics: dict[str, float] = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> None:
        """
        Executes the full training and evaluation pipeline.

        Args:
            X_train: The training features DataFrame.
            y_train: The training target labels.
            X_test: The test features DataFrame.
            y_test: The test target labels.
        """
        # The pipeline now handles column selection and transformation internally
        self.features.fit(X_train)
        X_train_t = self.features.transform(X_train)
        X_test_t = self.features.transform(X_test)

        # Train the model on the transformed training data
        self.model.fit(X_train_t, y_train)

        # Make predictions on the transformed test data
        preds = self.model.predict(X_test_t)

        # Calculate and store the accuracy metric
        self.metrics["accuracy"] = accuracy_score(y_test, preds)
        self._is_trained = True

    def get_metrics(self) -> dict[str, float]:
        """
        Returns the calculated performance metrics.
        """
        if not self._is_trained:
            raise RuntimeError("Trainer has not been run")
        return self.metrics
