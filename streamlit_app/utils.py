import streamlit.components.v1 as components
import pandas as pd
import pydeck as pdk
import streamlit as st

def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)


def create_layers(filtered_data):
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        filtered_data,
        get_position=["longitude", "latitude"],
        get_radius=10,
        get_fill_color=[255, 0, 0, 140],
        pickable=True,
    )
    return [scatter_layer]


def render_map(filtered_data):
    view_state = pdk.ViewState(
        latitude=filtered_data["latitude"].mean(),
        longitude=filtered_data["longitude"].mean(),
        zoom=4,
        pitch=0,
    )
    layers = create_layers(filtered_data,)
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v10",
        )
    )
