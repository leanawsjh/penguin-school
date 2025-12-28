import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class NumericScaler:
    """A wrapper for StandardScaler to handle numeric feature scaling."""
    def __init__(self):
        """Initializes the NumericScaler."""
        self._scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the scaler to the numeric data.

        Args:
            X: A NumPy array of numeric features.
        """
        self._scaler.fit(X)
        self._is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the numeric data using the fitted scaler.

        Args:
            X: A NumPy array of numeric features.

        Returns:
            The scaled numeric features.
        """
        if not self._is_fitted:
            raise RuntimeError("NumericScaler must be fitted first")
        return self._scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the scaler and transforms the data in one step.

        Args:
            X: A NumPy array of numeric features.

        Returns:
            The scaled numeric features.
        """
        self.fit(X)
        return self.transform(X)


class CategoricalEncoder:
    """A wrapper for OneHotEncoder to handle categorical feature encoding."""
    def __init__(self):
        """Initializes the CategoricalEncoder."""
        self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the encoder to the categorical data.

        Args:
            X: A NumPy array of categorical features.
        """
        self._encoder.fit(X)
        self._is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the categorical data using the fitted encoder.

        Args:
            X: A NumPy array of categorical features.

        Returns:
            The one-hot encoded categorical features.
        """
        if not self._is_fitted:
            raise RuntimeError("CategoricalEncoder must be fitted first")
        return self._encoder.transform(X)


class FeaturePipeline:
    """
    Orchestrates the entire feature engineering process.

    This pipeline takes numeric and categorical features, applies the appropriate
    scaling and encoding transformations, and combines them into a single
    feature matrix ready for model training.
    """
    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        numeric_scaler: NumericScaler,
        categorical_encoder: CategoricalEncoder,
    ):
        """
        Initializes the FeaturePipeline.

        Args:
            numeric_features: A list of column names for numeric features.
            categorical_features: A list of column names for categorical features.
            numeric_scaler: An instance of NumericScaler.
            categorical_encoder: An instance of CategoricalEncoder.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_scaler = numeric_scaler
        self.categorical_encoder = categorical_encoder

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePipeline":
        """
        Fits the numeric scaler and categorical encoder on the provided data.

        Args:
            X: The input DataFrame with features.
            y: The target series (optional, ignored).

        Returns:
            The fitted FeaturePipeline instance.
        """
        # Select data and convert to numpy arrays for fitting
        X_num = X[self.numeric_features].values
        X_cat = X[self.categorical_features].values

        self.numeric_scaler.fit(X_num)
        self.categorical_encoder.fit(X_cat)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input data using the fitted scalers and encoders.

        Args:
            X: The input DataFrame with features.

        Returns:
            A NumPy array containing the fully transformed and combined features.
        """
        # Select data and convert to numpy arrays for transforming
        X_num = X[self.numeric_features].values
        X_cat = X[self.categorical_features].values

        # Transform numeric and categorical features
        X_num_t = self.numeric_scaler.transform(X_num)
        X_cat_t = self.categorical_encoder.transform(X_cat)

        # Horizontally stack the transformed arrays
        return np.hstack([X_num_t, X_cat_t])