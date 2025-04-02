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

def preprocess_categorical(df):
    """
    Preprocess categorical columns in the DataFrame.
    """
    # Correct text in 'INFORMATION_SOURCE' column
    correction_map_is = {
        "Umowa ostateczna sprzedaĹĽy rynek wtĂłrny": "Umowa ostateczna sprzedaży rynek wtórny",
        "Ocena warto?ci zabezpieczenia na nieruchomo?ci": "Ocena wartości zabezpieczenia na nieruchomości",
        "Ocena wartości zabezpieczenia na nieruchomości (przed ESG)": "Ocena wartości zabezpieczenia na nieruchomości"
    }
    df['INFORMATION_SOURCE'] = df['INFORMATION_SOURCE'].replace(correction_map_is)

    # Correct text in 'BUILDING_KIND' column
    correction_map_bk = {
        "Wielorodzinny niski": "wielorodzinny",
        "Wielorodzinny wysoki": "wielorodzinny",
        "Budynek jednorodzinny": "jednorodzinny",
        "Inny": np.nan,
        "Brak informacji": np.nan
    }
    df['BUILDING_KIND'] = df['BUILDING_KIND'].replace(correction_map_bk)

    # Correct text in 'CONSTRUCTION_TYPE' column
    df['CONSTRUCTION_TYPE'] = df['CONSTRUCTION_TYPE'].replace({"MUROWANA (CEGŁA, PUSTAK)": "murowana"})

    # Unify information about missing values
    nan_replacements = {
        "TYPE_OF_BUILD": {"Brak informacji": np.nan},
        "BUILDING_TECHNICAL_CONDITION": {"ND": np.nan},
        "BUILDING_STANDARD_QUALITY": {"Brak informacji": np.nan},
        "PREMISSES_STANDARD_QUALITY": {"Brak informacji": np.nan},
        "VOIVODESHIP": {"INNE": np.nan}
    }

    for col, mapping in nan_replacements.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # Change 'sal_void_eoq' column datatype to int
    df['sal_void_eoq'] = pd.to_numeric(df['sal_void_eoq'], errors="coerce").astype("Int64")

    # Change text columns to lowercase
    cols_to_lower = ['PREMISSES_TECHNICAL_CONDITION', 'PREMISSES_STANDARD_QUALITY', 'VOIVODESHIP', 'COUNTY', 
                     'COMMUNITY', 'CITY', 'BUILDING_KIND', 'CONSTRUCTION_TYPE', 'TYPE_OF_BUILD', 'BUILDING_TECHNICAL_CONDITION', 
                     'BUILDING_STANDARD_QUALITY', 'voivodeship_flood_risk_rating', 'county_flood_risk_rating', 'INFORMATION_SOURCE']
    for col in cols_to_lower:
        if col in df.columns:
            df[col] = df[col].str.lower()

    return df

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

    # Preprocess categorical columns
    df_enriched = preprocess_categorical(df_enriched)

    return df_cleaned


