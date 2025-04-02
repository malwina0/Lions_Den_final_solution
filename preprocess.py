import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.impute import KNNImputer

def filter_out_non_housing(df):
    """
    Filter out rows that do not contain housing-related information.
    This function checks if the 'PROPERTY_KIND' column is either 'Lokal mieszkalny' or 'Budynek mieszkalny jednorodzinny'
    """
    return df[df['PROPERTY_KIND'].isin(['Lokal mieszkalny', 'Budynek mieszkalny jednorodzinny'])]


def drop_empty_columns(df):
    """
    Drop columns that have at least 60% of missing values and are not in COLS_TO_KEEP.
    """
    cols_to_drop = ['SUBTYPE', 'SHARE', 'UTIL_WATER_INTAKE', 'UTIL_WATER_SUPPLY', 'UTIL_ELECTRICITY',
                    'UTIL_GAS', 'UTIL_SEWAGE_SYSTEM_CONNECTION', 'UTIL_SEWAGE_SYSTEM_OWN', 'PREMISSES_STANDARD',
                    'PREMISSES_INDEX_PED', 'PREMISSES_INDEX_FED', 'PREMISSES_INDEX_UED', 'PREMISSES_ENERGY_PERF_CERT_DATE',
                    'PREMISSES_ENERGY_PERF_CERT_VALI', 'PREMISSES_RES_SHARE', 'PREMISSES_CO2_EMMISSION', 'CITY_ZONE',
                    'FLOORS_NO', 'BUILDING_CHAMBER_NO', 'BUILDING_INDEX_PED', 'BUILDING_INDEX_FED', 'BUILDING_INDEX_UED',
                    'BUILDING_ENERGY_PERF_CERT_DATE', 'BUILDING_ENERGY_PERF_CERT_VALI', 'BUILDING_RES_SHARE',
                    'BUILDING_CO2_EMMISSION', 'PARCEL_AREA', 'PARKING_SPACE_ID', 'PARKING_KIND', 'NUMBER']
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

def encode_categorical_columns(df):
    """
    Encode categorical columns ( using one-hot encoding or label encoding depending on the column values).
    """
    condition_mapping = { 'zły': 0,'średni': 1,'dobry': 2,'bardzo dobry': 3}
    df['PREMISSES_TECHNICAL_CONDITION'] = df['PREMISSES_TECHNICAL_CONDITION'].map(condition_mapping)
    df['BUILDING_TECHNICAL_CONDITION'] = df['BUILDING_TECHNICAL_CONDITION'].map(condition_mapping)

    building_quality_mapping = { 'niski': 0,'do remontu': 1,'średni': 2,'wysoki': 3, 'deweloperski / do wykończenia': 4}
    df['BUILDING_STANDARD_QUALITY'] = df['BUILDING_STANDARD_QUALITY'].map(building_quality_mapping)

    county_flood_risk_mapping = {'very low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
    df['county_flood_risk_rating'] = df['county_flood_risk_rating'].map(county_flood_risk_mapping)

    market_type_mapping = {'wtórny': 0, 'pierwotny': 1}
    df['MARKET_TYPE'] = df['MARKET_TYPE'].map(market_type_mapping)

    df = df[df['CONSTRUCTION_YEAR'] >= 1600]
    df = pd.get_dummies(df, columns = ['CONSTRUCTION_TYPE'], dtype=float)
    df = pd.get_dummies(df, columns = ['PROPERTY_KIND'], dtype=float)
    df = pd.get_dummies(df, columns = ['INFORMATION_SOURCE'], dtype=float)
    df = pd.get_dummies(df, columns = ['TYPE_OF_BUILD'], dtype=float)

    return df

def drop_columns(df):
    """
    Drop unnecessary columns from the DataFrame.
    """
    cols_to_drop = ['voivodeship_flood_risk_rating']
    df = df.drop(columns=cols_to_drop)

    return df

def fit_target_encoder(df, y):
    """
    Apply target encoder to 
    """
    city_encoder = TargetEncoder()
    df['CITY'] = city_encoder.fit_transform(df['CITY'], y)

    voivodeship_encoder = TargetEncoder()
    df['VOIVODESHIP'] = voivodeship_encoder.fit_transform(df['VOIVODESHIP'], y)

    county_encoder = TargetEncoder()
    df['COUNTY'] = county_encoder.fit_transform(df['COUNTY'], y)

    community_encoder = TargetEncoder()
    df['COMMUNITY'] = community_encoder.fit_transform(df['COMMUNITY'], y)

    return df, city_encoder, voivodeship_encoder, county_encoder, community_encoder

def apply_target_encoder(df, city_encoder, voivodeship_encoder, county_encoder, community_encoder):
    """
    Apply target encoder to the DataFrame.
    Note: fit(X, y).transform(X) does not equal fit_transform(X, y).
    """
    df['CITY'] = city_encoder.transform(df['CITY'])
    df['VOIVODESHIP'] = voivodeship_encoder.transform(df['VOIVODESHIP'])
    df['COUNTY'] = county_encoder.transform(df['COUNTY'])
    df['COMMUNITY'] = community_encoder.transform(df['COMMUNITY'])

    return df

def custom_imputation(df, knn_neighbors=5):
    """
    Custom imputation function for missing values in the DataFrame.
    - Imputes "Nie" for specific columns.
    - Imputes -1 for columns with >= 25% missing values.
    - Applies KNN imputation for columns with < 25% missing values.
    """
    df = df.copy()  
    special_impute_cols = ['RENEWABLE_ENERGY_HEATING', 'RENEWABLE_ENERGY_ELECTRIC', 'CERTIFICATE_PHI']
    
    # Impute "Nie" for special columns
    for col in special_impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Nie")
    
    missing_perc = df.isna().mean() * 100

    # Impute -1 for columns with >= 25% missing values
    high_missing_cols = missing_perc[missing_perc >= 25].index.tolist()
    df[high_missing_cols] = df[high_missing_cols].fillna(-1)
    
    # KNN Imputation for columns with < 25% missing values
    low_missing_cols = missing_perc[(missing_perc > 0) & (missing_perc < 25)].index.tolist()
    if low_missing_cols:
        knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
        imputed_df = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns, index=df.index)

    return imputed_df, high_missing_cols, low_missing_cols, knn_imputer

def apply_custom_imputation(df, high_missing_cols, low_missing_cols, knn_imputer):
    """
    Perform custom imputation on the test set.
    """
    df = df.copy()  
    special_impute_cols = ['RENEWABLE_ENERGY_HEATING', 'RENEWABLE_ENERGY_ELECTRIC', 'CERTIFICATE_PHI']
    
    for col in special_impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Nie")
    
    df[high_missing_cols] = df[high_missing_cols].fillna(-1)
    if low_missing_cols:
        imputed_df = pd.DataFrame(knn_imputer.transform(df), columns=df.columns, index=df.index)

    return imputed_df

def preprocess_train(X, y, df_me):
    """
    Preprocess the DataFrame by filtering out non-housing data and dropping empty columns.
    """
    # Filter out non-housing data
    df_housing = filter_out_non_housing(X)

    # Drop empty columns
    df_cleaned = drop_empty_columns(df_housing)

    # encode categorical columns
    df_encoded = encode_categorical_columns(df_cleaned)

    # Drop unnecessary columns
    df_cleaned_encoded = drop_columns(df_encoded)

    # Enrich the DataFrame with ME transformation data
    df_enriched = enrich_dataframe_with_me_transformation(df_cleaned_encoded, df_me)

    # Preprocess categorical columns
    df_enriched = preprocess_categorical(df_enriched)

    # Fit and apply target encoder to the training set
    df_enriched, city_encoder, voivodeship_encoder, county_encoder, community_encoder = fit_target_encoder(df_enriched, y)

    # Fit and apply imputation
    df_imputed, high_missing_cols, low_missing_cols, knn_imputer = custom_imputation(df_enriched)

    return df_imputed, city_encoder, voivodeship_encoder, county_encoder, community_encoder, high_missing_cols, low_missing_cols, knn_imputer 


def preprocess_test(X, df_me, city_encoder, voivodeship_encoder, county_encoder, community_encoder, high_missing_cols, low_missing_cols, knn_imputer):
    """
    Preprocess the DataFrame by filtering out non-housing data and dropping empty columns.
    """
    # Filter out non-housing data
    df_housing = filter_out_non_housing(X)

    # Drop empty columns
    df_cleaned = drop_empty_columns(df_housing)

    # encode categorical columns
    df_encoded = encode_categorical_columns(df_cleaned)

    # Drop unnecessary columns
    df_cleaned_encoded = drop_columns(df_encoded)

    # Enrich the DataFrame with ME transformation data
    df_enriched = enrich_dataframe_with_me_transformation(df_cleaned_encoded, df_me)

    # Preprocess categorical columns
    df_enriched = preprocess_categorical(df_enriched)

    # Apply target encoder to the test set
    df_enriched = apply_target_encoder(df_enriched, city_encoder, voivodeship_encoder, county_encoder, community_encoder)

    # Apply imputation
    df_imputed = apply_custom_imputation(df_enriched, high_missing_cols, low_missing_cols, knn_imputer)

    return df_imputed











