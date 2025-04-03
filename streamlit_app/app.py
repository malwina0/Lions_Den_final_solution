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

st.set_page_config(layout="wide")

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
with open('streamlit_app/train_test_data2.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

# Load model
with open("streamlit_app/best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("streamlit_app/shap_explainer.pkl", "rb") as explainer_file:
    explainer = pickle.load(explainer_file)

with open("streamlit_app/shap_values.pkl", "rb") as shap_file:
    shap_values = pickle.load(shap_file)

def main():
    st.title('Analiza zależności od VALUATION_VALUE')

    # Ładowanie danych
    df = load_data()

    # Tworzenie zakładek
    tab1, tab2, tab3, tab4 = st.tabs(["Training Data Visualizations", "Marco-economic relationships", "correlation matrix", "Linear regression" ])

    with tab1:
        st.subheader('Training Data Visualizations')
        dashboard.show(X_train, y_train, X_test, y_test, model, explainer, shap_values, "#FFFFFF")
    # Zakładka 1 - Wykresy zależności
    with tab2:
        st.subheader('Relationships in data')
        create_scatter_plot(df)
        plot_correlation_matrix(df)

    # Zakładka 2 - Macierz korelacji
    with tab3:
        st.subheader('Correlation matrix')
        plot_correlation_matrix(df)

    with tab4:
        multiple_linear_regression(df, 'VALUATION_VALUE')

if __name__ == "__main__":
    main()