import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw penguin dataset by handling missing values and correcting inconsistencies.
    """
    df = df.dropna()
    df['sex'] = df['sex'].replace('.', pd.NA).dropna()
    return df
