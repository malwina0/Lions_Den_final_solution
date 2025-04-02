import pandas as pd
import numpy as np

def filter_out_non_housing(df):
    """
    Filter out rows that do not contain housing-related information.
    This function checks if the 'PROPERTY_KIND' column is either 'Lokal mieszkalny' or 'Budynek mieszkalny jednorodzinny'
    """
    return df[df['PROPERTY_KIND'].isin(['Lokal mieszkalny', 'Budynek mieszkalny jednorodzinny'])]


def drop_empty_columns(df, threshold=0.6):
    """
    Drop columns that have at least 60% of missing values and are not in COLS_TO_KEEP.
    """
    COLS_TO_KEEP = ['RENEWABLE_ENERGY_HEATING', 'RENEWABLE_ENERGY_ELECTRIC', 'CERTIFICATE_PHI', 'STOREY']
    missing_ratio = df.isnull().mean()
    cols_to_drop = [col for col in df.columns if (missing_ratio[col] >= threshold and col not in COLS_TO_KEEP)]
    df_cleaned = df.drop(columns=cols_to_drop)

    return df_cleaned


def preprocess(df):
    """
    Preprocess the DataFrame by filtering out non-housing data and dropping empty columns.
    """
    # Filter out non-housing data
    df_housing = filter_out_non_housing(df)

    # Drop empty columns
    df_cleaned = drop_empty_columns(df_housing)

    return df_cleaned


