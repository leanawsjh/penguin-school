
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from src.pipelines.data import load_data
from src.pipelines.config import TrainConfig

@pytest.fixture
def mock_penguin_data():
    """Fixture to create a sample DataFrame for mocking."""
    data = {
        'culmen_length_mm': [39.1, 49.0],
        'culmen_depth_mm': [18.7, 16.1],
        'flipper_length_mm': [181.0, 220.0],
        'body_mass_g': [3750.0, 5200.0],
        'species': ['Adelie', 'Gentoo'],
        'island': ['Torgersen', 'Biscoe'],
        'sex': ['MALE', 'FEMALE']
    }
    return pd.DataFrame(data)

@patch('pandas.read_csv')
def test_load_data(mock_read_csv, mock_penguin_data):
    """Test the load_data function."""
    mock_read_csv.return_value = mock_penguin_data

    features, target = load_data()

    # Check the types
    assert isinstance(features, np.ndarray)
    assert isinstance(target, np.ndarray)

    config = TrainConfig()
    # Check the shapes
    assert features.shape == (2, len(config.num_columns + config.cat_columns))
    assert target.shape == (2,)

    # Check the content
    expected_features = mock_penguin_data[config.num_columns + config.cat_columns].to_numpy()
    expected_target = mock_penguin_data[config.target].to_numpy()

    np.testing.assert_array_equal(features, expected_features)
    np.testing.assert_array_equal(target, expected_target)

    # Verify that read_csv was called correctly
    config = TrainConfig()
    mock_read_csv.assert_called_once_with(config.data_path)
