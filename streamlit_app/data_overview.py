import streamlit as st

def show(data):
    st.markdown("""
        <h3 style='color: #ff6200;'>Data overview</h3>
    """, unsafe_allow_html=True)
    
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
