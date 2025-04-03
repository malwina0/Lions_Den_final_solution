import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import ColourWidgetText, styled_subheader
import shap
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats


ING_COLOR = "#ff6200"

def show(X_train, y_train, X_test, y_test, model, shap_values, COLOR):  
    # y_train_original = np.expm1(y_train)
    # y_test_original = np.expm1(y_test)  
    sorted_indices_train = y_train.sort_values(ascending=True).index
    sorted_indices_test = y_test.sort_values(ascending=True).index
    sorted_predictions_train = model.predict(X_train)[sorted_indices_train]
    sorted_predictions_test = model.predict(X_test)[sorted_indices_test]

    styled_subheader("Data Statistics")
    col1, col2, col3 = st.columns(3)

    # # Ensure predictions are writable by creating copies
    # train_predictions = model.predict(X_train).copy()
    # test_predictions = model.predict(X_test).copy()
    sorted_train = y_train[sorted_indices_train]
    sorted_test = y_test[sorted_indices_test]
    sorted_predictions_train = model.predict(X_train)[sorted_indices_train]
    sorted_predictions_test = model.predict(X_test)[sorted_indices_test]

    # Calculate metrics
    train_mae = mean_absolute_error(np.expm1(sorted_predictions_train), sorted_train)
    test_mae = mean_absolute_error(np.expm1(sorted_predictions_test), sorted_test)

    # Display metrics
    col1.metric("Observations number", f"{len(X_test)+len(X_test)}")
    col2.metric("Train Mean Absolute Error", f"{train_mae:.2f}")
    col3.metric("Test Mean Absolute Error", f"{test_mae:.2f}")

    ColourWidgetText(f"{len(X_test)+len(X_test)}", '#808080') 
    ColourWidgetText(f"{train_mae:.2f}", '#808080') 
    ColourWidgetText(f"{test_mae:.2f}", '#808080')

    # st.markdown("### Train data ")
    col1, col2 = st.columns(2)
    with col1:
        colors = px.colors.sequential.Inferno
        styled_subheader("Predictions Train")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(y_train)), y=np.expm1(sorted_predictions_train), mode='lines', name='Predictions Train', line=dict(color=colors[4])))
        fig1.add_trace(go.Scatter(x=np.arange(len(y_train)), y=y_train.sort_values(ascending=True).values, mode='lines', name='Labels Train', line=dict(color=colors[7])))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        styled_subheader("Predictions Test")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(y_test)), y=np.expm1(sorted_predictions_test), mode='lines', name='Predictions Test', line=dict(color=colors[4])))
        fig1.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.sort_values(ascending=True).values, mode='lines', name='Labels Test', line=dict(color=colors[7])))
        st.plotly_chart(fig1, use_container_width=True)

    # SHAP PLOTS!!

    
    col1, col2 = st.columns(2)
    axes_color = "gray"

    with col1:
        styled_subheader("SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", cmap = "OrRd")  # Generate the SHAP plot
        # Customize the figure to match Streamlit's dark theme
        fig.patch.set_alpha(0.0)  # Set figure background color
        w, h = fig.get_size_inches()
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
        fig, ax = plt.subplots(figsize=(w, h))
        shap.plots.bar(shap_values, show=False, max_display=shap_values.shape[1])  # Generate the SHAP bar plot

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



def load_data():
    # Załaduj dane do DataFrame (wstaw swoją ścieżkę do pliku)
    df = pd.read_csv('Me_summary.csv')
    df['year_month'] = pd.to_datetime(df['year_month'], errors='coerce')
    return df


# Funkcja do tworzenia wykresu zależności
def create_scatter_plot(df):
    # Wykres zależności 'VALUATION_VALUE' od '3MR_Annual_absolute_difference'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['3MR_Annual_absolute_difference'], y=df['VALUATION_VALUE'], ax=ax)
    ax.set_title('Relationship VALUATION_VALUE and 3MR_Annual_absolute_difference')
    ax.set_xlabel('3MR_Annual_absolute_difference')
    ax.set_ylabel('VALUATION_VALUE')

    # Zwracamy wykres
    st.pyplot(fig)

    # Wykres zależności 'VALUATION_VALUE' od '3MR_Raw'
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['3MR_Raw'], y=df['VALUATION_VALUE'], ax=ax)
    ax.set_title('Zależność VALUATION_VALUE od 3MR_Raw')
    ax.set_xlabel('3MR_Raw')
    ax.set_ylabel('VALUATION_VALUE')

    # Zwracamy wykres
    st.pyplot(fig)


# Funkcja do wyświetlania macierzy korelacji
def plot_correlation_matrix(df):
    df_tmp = df.drop(columns=['year_month']).columns
    correlation_matrix = df_tmp.corr()

    # Wykres macierzy korelacji
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation matrix')

    # Wyświetlenie wykresu
    st.pyplot(fig)

def multiple_linear_regression(df, target_variable):
    # Select all columns except the target variable and date columns
    feature_columns = df.drop(columns=[target_variable, 'year_month']).columns
    X = df[feature_columns]  # Independent variables (features)
    y = df[target_variable]  # Dependent variable (target)

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()
    st.write(model.summary())  # Display regression results

    # Visualization of multiple regression (using one feature for simplicity)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df['3MR_Annual_absolute_difference'], y=df[target_variable], ax=ax, scatter_kws={'s': 50}, line_kws={'color': 'blue'})
    ax.set_title(f'Multiple Linear Regression: 3MR_Annual_absolute_difference vs {target_variable}')
    ax.set_xlabel('3MR_Annual_absolute_difference')
    ax.set_ylabel(target_variable)
    st.pyplot(fig)


def multiple_linear_regression(df, target_variable):
    # Select all columns except the target variable and date columns
    feature_columns = df.drop(columns=[target_variable, 'year_month']).columns
    X = df[feature_columns]  # Independent variables (features)
    y = df[target_variable]  # Dependent variable (target)

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()
    st.write(model.summary())  # Display regression results

    # Visualization of multiple regression (using one feature for simplicity)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df['3MR_Annual_absolute_difference'], y=df[target_variable], ax=ax, scatter_kws={'s': 50}, line_kws={'color': 'blue'})
    ax.set_title(f'Multiple Linear Regression: 3MR_Annual_absolute_difference vs {target_variable}')
    ax.set_xlabel('3MR_Annual_absolute_difference')
    ax.set_ylabel(target_variable)
    st.pyplot(fig)
