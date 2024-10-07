import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_simple_regression(X, y, model, X_test, y_test):
    # Make sure to predict using the training data if num_features is 1
    if X.shape[1] == 1:
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(X_test)

    df = pd.DataFrame({'X': X.flatten(), 'y': y, 'y_pred': y_pred[:len(y)]})

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='X', y='y', label='Data Points', color='blue', alpha=0.6)
    sns.lineplot(data=df, x='X', y='y_pred', label='Regression Line', color='red', linewidth=2)
    
    plt.axhline(y=np.mean(y), color='orange', linestyle='--', label='Mean Line')
    plt.title('Simple Linear Regression', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Target Variable', fontsize=14)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    st.pyplot()

def plot_3d_regression(X, y, model):
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    
    # Calculate regression surface
    y_grid = model.intercept_ + model.coef_[0] * x1_grid + model.coef_[1] * x2_grid

    # Calculate the mean Z value for the mean plane
    mean_y = model.intercept_ + model.coef_[0] * np.mean(X[:, 0]) + model.coef_[1] * np.mean(X[:, 1])
    mean_plane = np.full_like(x1_grid, mean_y)  # Create an array filled with the mean value

    fig = go.Figure()

    # Add scatter plot for actual data
    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.6),
        name='Data Points'
    ))

    # Add regression surface
    fig.add_trace(go.Surface(
        x=x1_grid, y=x2_grid, z=y_grid,
        colorscale='Viridis', opacity=0.5, name='Regression Plane', showlegend=True
    ))

    # Add mean plane
    fig.add_trace(go.Surface(
        x=x1_grid, y=x2_grid, z=mean_plane,
        colorscale='Reds', opacity=0.5, name='Mean Plane', showscale=False, showlegend=True
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Target Variable',
            camera=dict(
                eye=dict(x=1.5, y=1.4, z=0.8)  # Adjust for the desired view
            )
        ),
        title='3D Multiple Linear Regression with Mean Plane',
        height=700,
        showlegend=True,  # Enable the legend
        legend=dict(x=0.1, y=0.9)  # Position the legend
    )
    
    st.plotly_chart(fig)

def plot_pair_plot(df):
    st.subheader("Pair Plots")
    fig = px.scatter_matrix(df)
    fig.update_layout(title='Pair Plot')
    st.plotly_chart(fig)

def plot_actual_vs_predicted(y_test, y_pred):
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig = px.scatter(df, x='Actual', y='Predicted', title='Actual vs Predicted',
                     labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
                     trendline='ols')
    
    fig.add_hline(y=0, line_color='red', line_dash='dash')
    st.plotly_chart(fig)

def plot_residuals(y_pred, residuals):
    df = pd.DataFrame({'Predicted': y_pred, 'Residuals': residuals})
    fig = px.scatter(df, x='Predicted', y='Residuals', title='Residuals Plot',
                     labels={'Predicted': 'Predicted Values', 'Residuals': 'Residuals'})
    
    fig.add_hline(y=0, line_color='black', line_dash='dash')
    st.plotly_chart(fig)

def plot_histogram_of_residuals(residuals):
    fig = px.histogram(x=residuals, title='Histogram of Residuals', 
                       labels={'x': 'Residuals'}, nbins=30, 
                       color_discrete_sequence=['skyblue'])
    st.plotly_chart(fig)

def plot_hat_matrix(hat_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=hat_matrix,
        colorscale='Viridis',
        colorbar=dict(title='Leverage'),
    ))
    fig.update_layout(title='Hat Matrix (Leverage)', xaxis_title='Features', yaxis_title='Observations')
    st.plotly_chart(fig)

def plot_beta_cap(beta_cap):
    df = pd.DataFrame({'Features': [f'Feature {i+1}' for i in range(len(beta_cap))], 'Coefficients': beta_cap})
    fig = px.bar(df, x='Features', y='Coefficients', title='Estimated Coefficients (Beta Cap)',
                 labels={'Coefficients': 'Coefficient Value'})
    st.plotly_chart(fig)

def plot_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Features"] = [f'Feature {i+1}' for i in range(X.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

    fig = px.bar(vif_data, x='VIF', y='Features', title='Variance Inflation Factor (VIF)', 
                  labels={'VIF': 'VIF', 'Features': 'Features'},
                  color='VIF', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
