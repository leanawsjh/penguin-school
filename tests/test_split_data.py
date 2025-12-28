
import numpy as np
import pytest
from src.pipelines.data import split_data
from src.pipelines.config import TrainConfig

@pytest.fixture
def sample_data():
    """Fixture to create sample data for splitting."""
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    return X, y

def test_split_data(sample_data):
    """Test the split_data function."""
    X, y = sample_data
    config = TrainConfig()

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    # Check the types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # Check the shapes based on the test_size
    from math import ceil
    expected_test_size = int(ceil(X.shape[0] * config.test_size))
    expected_train_size = X.shape[0] - expected_test_size

    assert X_train.shape == (expected_train_size, X.shape[1])
    assert X_test.shape == (expected_test_size, X.shape[1])
    assert y_train.shape == (expected_train_size,)
    assert y_test.shape == (expected_test_size,)

    # Check that the data is shuffled
    assert not np.array_equal(np.concatenate([y_train, y_test]), y)
