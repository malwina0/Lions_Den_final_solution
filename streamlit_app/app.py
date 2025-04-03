import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost
import pickle
from sklearn.metrics import mean_absolute_error
import streamlit.components.v1 as components
import pydeck as pdk
import dashboard
import map
import data_overview
import matplotlib.pyplot as plt
import shap
from dashboard import *

# color palette: https://pythonfriday.dev/2023/09/193-choosing-colours-for-plotly/, #8C2972

# Set Streamlit dark theme

# orange palette: #ff6200, rgba(255,98,0,255)

COLOR = "#FFFFFF"
METRIC_COLOR = "#ff6200"

# st.set_page_config(layout="wide")

st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: {COLOR};  
        color: white;
    }}
    .stMetric {{
        font-size: 24px;
        font-weight: bold;
        color: {METRIC_COLOR};
    }}
    .css-1d391kg {{
        background-color: {COLOR} !important;  
    }}
    .stTabs {{
        font-size: 20px !important;
        font-weight: bold;
    }}
    .stColumns > div {{
        flex: 1;
        padding: 30px;
    }}
    </style>
""", unsafe_allow_html=True)

# Load data
with open("streamlit_app/dataset.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Load model
with open("streamlit_app/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

def main():
    st.title('Analiza zależności od VALUATION_VALUE')

    # Ładowanie danych
    df = load_data()

    # Tworzenie zakładek
    tab1, tab2, tab3, tab4 = st.tabs(["Marco-economic relationships", "correlation matrix", "Linear regression", "General relationships"])

    # Zakładka 1 - Wykresy zależności
    with tab1:
        plot_scatters()

    # Zakładka 2 - Macierz korelacji
    with tab2:
        st.subheader('Correlation matrix')
        plot_correlation_matrix(df)

    with tab3:
        multiple_linear_regression(df, 'VALUATION_VALUE')

    with tab4:
        st.subheader('Relationships in data')
        create_scatter_plot(df)
        plot_correlation_matrix(df)

if __name__ == "__main__":
    main()