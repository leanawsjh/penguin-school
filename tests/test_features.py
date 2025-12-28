import pandas as pd
from src.pipelines.features import engineer_features

def test_engineer_features_one_hot_encoding():
    # Test case: DataFrame with categorical features for one-hot encoding
    data = {'species': ['Adelie', 'Chinstrap', 'Gentoo'],
            'island': ['Torgersen', 'Dream', 'Biscoe'],
            'culmen_length_mm': [39.1, 50.0, 46.1],
            'sex': ['MALE', 'FEMALE', 'MALE']}
    df = pd.DataFrame(data)
    engineered_df = engineer_features(df)

    # Check if original categorical columns are dropped
    assert 'island' not in engineered_df.columns
    assert 'sex' not in engineered_df.columns

    # Check for the presence of new one-hot encoded columns (example for 'island' and 'sex')
    assert 'island_Torgersen' in engineered_df.columns
    assert 'island_Dream' in engineered_df.columns
    assert 'island_Biscoe' in engineered_df.columns
    assert 'sex_MALE' in engineered_df.columns
    assert 'sex_FEMALE' in engineered_df.columns

    # Check the values of one-hot encoded columns for a specific row
    # For the first row (Adelie, Torgersen, MALE)
    assert engineered_df.loc[0, 'island_Torgersen'] == 1
    assert engineered_df.loc[0, 'island_Dream'] == 0
    assert engineered_df.loc[0, 'island_Biscoe'] == 0
    assert engineered_df.loc[0, 'sex_MALE'] == 1
    assert engineered_df.loc[0, 'sex_FEMALE'] == 0

