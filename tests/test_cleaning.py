import pandas as pd
from src.pipelines.cleaning import clean_data

def test_clean_data_missing_values():
    # Test case: DataFrame with missing values
    data = {'species': ['Adelie', 'Adelie', 'Adelie', 'Adelie'],
            'island': ['Torgersen', 'Torgersen', 'Torgersen', 'Torgersen'],
            'culmen_length_mm': [39.1, 39.5, 40.3, pd.NA],
            'culmen_depth_mm': [18.7, 17.4, 18.0, pd.NA],
            'flipper_length_mm': [181, 186, 195, pd.NA],
            'body_mass_g': [3750, 3800, 3250, pd.NA],
            'sex': ['MALE', 'FEMALE', 'FEMALE', pd.NA]}
    df = pd.DataFrame(data)
    cleaned_df = clean_data(df)
    assert not cleaned_df.isnull().any().any(), "DataFrame should not have any missing values after cleaning"

def test_clean_data_sex_inconsistency():
    # Test case: DataFrame with inconsistent 'sex' values
    data = {'species': ['Gentoo', 'Gentoo', 'Gentoo'],
            'island': ['Biscoe', 'Biscoe', 'Biscoe'],
            'culmen_length_mm': [44.5, 48.8, 47.2],
            'culmen_depth_mm': [15.7, 16.2, 13.7],
            'flipper_length_mm': [217, 222, 214],
            'body_mass_g': [4875, 6000, 4925],
            'sex': ['FEMALE', 'MALE', '.']}
    df = pd.DataFrame(data)
    cleaned_df = clean_data(df)
    assert all(sex in ['MALE', 'FEMALE'] for sex in cleaned_df['sex'].dropna().unique()), "'sex' column should only contain 'MALE' or 'FEMALE'"
