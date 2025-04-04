import streamlit as st
from utils import styled_subheader

def show(data):
    
    styled_subheader("Data overview")
    
    # Użycie kontenera dla obramowania
    with st.container():
        st.markdown("""
            <style>
                div[data-testid="stDataFrame"] {
                    border: 2px solid #ff6200;
                    border-radius: 5px;
                    padding: 5px;
                }
                thead tr th {
                    background-color: #ff6200 !important;
                    color: white !important;
                }
            </style>
        """, unsafe_allow_html=True)

        st.dataframe(data, height=400, use_container_width=True)
