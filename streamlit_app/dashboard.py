import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import ColourWidgetText, styled_subheader
import shap
from matplotlib import pyplot as plt

ING_COLOR = "#ff6200"

def show(X_train, y_train, X_test, y_test, model, COLOR):    
    sorted_indices_train = y_train.sort_values(ascending=True).index
    sorted_indices_test = y_test.sort_values(ascending=True).index
    sorted_predictions_train = model.predict(X_train)[sorted_indices_train]
    sorted_predictions_test = model.predict(X_test)[sorted_indices_test]

    # st.markdown("### Train data ")
    col1, col2 = st.columns(2)
    with col1:
        colors = px.colors.sequential.Inferno
        styled_subheader("Predictions Train")

        sorted_indices_train = y_train.sort_values(ascending=True).index
        sorted_indices_test = y_test.sort_values(ascending=True).index
        sorted_predictions_train = model.predict(X_train)[sorted_indices_train]
        sorted_predictions_test = model.predict(X_test)[sorted_indices_test]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(y_train)), y=sorted_predictions_train, mode='lines', name='Predictions Train', line=dict(color=colors[4])))
        fig1.add_trace(go.Scatter(x=np.arange(len(y_train)), y=y_train.sort_values(ascending=True).values, mode='lines', name='Predictions Train', line=dict(color=colors[7])))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        styled_subheader("Predictions Test")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(y_test)), y=sorted_predictions_test, mode='lines', name='Predictions Train', line=dict(color=colors[4])))
        fig1.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.sort_values(ascending=True).values, mode='lines', name='Predictions Train', line=dict(color=colors[7])))
        st.plotly_chart(fig1, use_container_width=True)

    # SHAP PLOTS!!

    
    col1, col2 = st.columns(2)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    axes_color = "gray"

    with col1:
        styled_subheader("SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", cmap = "OrRd")  # Generate the SHAP plot
        # Customize the figure to match Streamlit's dark theme
        fig.patch.set_alpha(0.0)  # Set figure background color
        ax = plt.gca()  # Get the current axes
        ax.set_facecolor(COLOR)  # Set axes background color
        ax.title.set_color(axes_color)  # Set title color to white
        ax.xaxis.label.set_color(axes_color)  # Set x-axis label color to white
        ax.yaxis.label.set_color(axes_color)  # Set y-axis label color to white
        ax.tick_params(colors=axes_color)  # Set tick label colors to white

        # Display the plot in Streamlit
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        styled_subheader("SHAP Bar Plot")
        # Generate the SHAP bar plot
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)  # Generate the SHAP bar plot

        for bar in ax.patches:  # Iterate through all bars in the plot
            bar.set_facecolor(ING_COLOR)  # Set the bar color to orange
        
        for text in ax.texts:  # Iterate through all text labels in the plot
            text.set_color("gray")  # Set the text color to gray
            text.set_fontsize(8)
            

        # Customize the figure to match Streamlit's theme
        fig.patch.set_alpha(0.0)  # Set figure background color
        ax = plt.gca()  # Get the current axes
        ax.set_facecolor(COLOR)  # Set axes background color
        ax.title.set_color(axes_color)  # Set title color to white
        ax.xaxis.label.set_color(axes_color)  # Set x-axis label color to white
        ax.yaxis.label.set_color(axes_color)  # Set y-axis label color to white
        ax.tick_params(colors=axes_color)  # Set tick label colors to white

        st.pyplot(fig, use_container_width=True)


    styled_subheader("Data Statistics")
    col1, col2, col3 = st.columns(3)

    # Ensure predictions are writable by creating copies
    train_predictions = model.predict(X_train).copy()
    test_predictions = model.predict(X_test).copy()

    # Calculate metrics
    train_mae = mean_absolute_error(train_predictions, y_train)
    test_mae = mean_absolute_error(test_predictions, y_test)

    # Display metrics
    col1.metric("Observations number", f"{len(X_test)}")
    col2.metric("Train Mean Absolute Error", f"{train_mae:.2f}")
    col3.metric("Test Mean Absolute Error", f"{test_mae:.2f}")

    ColourWidgetText(f"{len(X_test)}", '#808080') 
    ColourWidgetText(f"{train_mae:.2f}", '#808080') 
    ColourWidgetText(f"{test_mae:.2f}", '#808080') 