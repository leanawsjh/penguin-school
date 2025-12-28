import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np

class XGBoostModel:
    """A wrapper for the XGBoost Classifier model."""
    def __init__(self, **kwargs):
        """
        Initializes the XGBoostModel.

        Args:
            **kwargs: Keyword arguments to be passed to the xgb.XGBClassifier.
        """
        self.model = xgb.XGBClassifier(**kwargs)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the XGBoost model to the training data.

        Args:
            X: The training feature data.
            y: The training target data.
        """
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: np.ndarray):
        """
        Makes predictions on the input data using the fitted model.

        Args:
            X: The input feature data for prediction.

        Returns:
            A NumPy array of predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray):
        """
        Predicts class probabilities for the input data.

        Args:
            X: The input feature data for prediction.

        Returns:
            A NumPy array of class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, cv=5):
        """
        Evaluates the model using cross-validation.

        This method uses scikit-learn's cross_val_score to perform k-fold
        cross-validation on the provided data.

        Args:
            X: The feature data for evaluation.
            y: The target data for evaluation.
            cv: The number of cross-validation folds.

        Returns:
            A NumPy array of scores for each fold.
        """
        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores