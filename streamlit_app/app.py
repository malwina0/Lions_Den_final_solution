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
with open("dataset.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)


# Tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Map", "Data overview"])

with tab1:
    dashboard.show(X_train, y_train, X_test, y_test, model, COLOR)

# with tab2:
#     data = pd.read_csv("../processed_data.csv")
#     map.show(data)

with tab3:
    data_overview.show(X_train)