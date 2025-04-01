import streamlit as st
import pandas as pd
from utils import render_map

def show(data):
    render_map(data)
    st.write(
        f"Search results: {data.shape[0]}"
    )
