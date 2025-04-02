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


def enrich_dataframe_with_me_transformation(df: pd.DataFrame, df_me: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the given DataFrame `df` by dynamically adding columns containing
    values from `df_me` based on unique pairs of ME_Indicator and Transformation.

    Parameters:
    - df (pd.DataFrame): The main DataFrame with a 'VALUE_DATE' column.
    - df_me (pd.DataFrame): The reference DataFrame containing ME_Indicator, Transformation values,
      and associated time periods.

    Returns:
    - pd.DataFrame: The enriched DataFrame with new columns for each ME_Indicator-Transformation pair.
    """

    # Convert date columns to datetime format
    df_me['Date_period_from'] = pd.to_datetime(df_me['Date_period_from'], format='%Y-%m-%d')
    df_me['Date_period_until'] = pd.to_datetime(df_me['Date_period_until'], format='%Y-%m-%d')

    df['VALUE_DATE_ORIGINAL'] = df['VALUE_DATE']
    df['VALUE_DATE'] = pd.to_datetime(df['VALUE_DATE'], format='%Y-%m-%d', errors='coerce')

    # Drop rows where VALUE_DATE is NaN - few values are dropped - explained in report
    df = df.dropna(subset=['VALUE_DATE']).reset_index(drop=True)

    # Create new columns based on ME_Indicator and Transformation pairs
    for indicator, transformation in df_me[['ME_Indicator', 'Transformation']].drop_duplicates().values:
        column_name = f"{indicator}_{transformation}".replace(" ", "_")

        df[column_name] = df['VALUE_DATE'].apply(
            lambda date: df_me.loc[
                (df_me['ME_Indicator'] == indicator) &
                (df_me['Transformation'] == transformation) &
                (df_me['Date_period_from'] <= date) &
                (df_me['Date_period_until'] >= date), 'Value'
            ].values[0]
            if not df_me.loc[
                (df_me['ME_Indicator'] == indicator) &
                (df_me['Transformation'] == transformation) &
                (df_me['Date_period_from'] <= date) &
                (df_me['Date_period_until'] >= date)
                ].empty else None
        )

    return df


def preprocess(df, df_me):
    """
    Preprocess the DataFrame by filtering out non-housing data and dropping empty columns.
    """
    # Filter out non-housing data
    df_housing = filter_out_non_housing(df)

    # Drop empty columns
    df_cleaned = drop_empty_columns(df_housing)

    # Enrich the DataFrame with ME transformation data
    df_enriched = enrich_dataframe_with_me_transformation(df_cleaned, df_me)

    return df_cleaned


