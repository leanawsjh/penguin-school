from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import the project's configuration and cleaning function
from .config import TrainConfig
from .cleaning import clean_data


def get_cleaned_data(config: TrainConfig) -> pd.DataFrame:
    """
    Load the raw penguin dataset and apply cleaning steps.

    Args:
        config: The project's training configuration object.

    Returns:
        A cleaned pandas DataFrame.
    """
    raw_df = pd.read_csv(config.data_path)
    return clean_data(raw_df)


def encode_target(y: pd.Series) -> pd.Series:
    """
    Encode the target variable using LabelEncoder.

    Args:
        y: The target variable series.

    Returns:
        The encoded target variable series with the encoder attached.
    """
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), index=y.index)
    y_encoded.attrs['le'] = le
    return y_encoded


def split_data(
    df: pd.DataFrame, config: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the cleaned data into training and testing sets.

    Args:
        df: The cleaned pandas DataFrame.
        config: The project's training configuration object.

    Returns:
        A tuple containing X_train, X_test, y_train, y_test.
    """
    X = df.drop(config.target, axis=1)
    y = df[config.target]

    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
