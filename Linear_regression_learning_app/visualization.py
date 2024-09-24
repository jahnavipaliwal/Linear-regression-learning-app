# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_line, geom_hline, labs, theme

def plot_simple_regression(X, y, model, X_test, y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Data Points')
    
    y_fit = model.predict(X_test)
    plt.plot(X_test, y_fit, color='red', linewidth=2, label='Regression Line')
    
    mean_y = np.mean(y)
    plt.axhline(mean_y, color='orange', linestyle='--', label='Mean Line')

    for x_val, y_actual, y_predicted in zip(X_test, y_test, y_pred):
        plt.plot([x_val, x_val], [y_predicted, y_actual], color='green', linestyle=':', linewidth=2, label='SSE Distance' if x_val == X_test[0] else "")
        plt.plot([x_val, x_val], [mean_y, y_predicted], color='purple', linestyle='--', linewidth=2, label='SSR Distance' if x_val == X_test[0] else "")

    plt.title('Simple Linear Regression')
    plt.xlabel('Feature 1')
    plt.ylabel('Target Variable')
    plt.legend()
    plt.grid()
    st.pyplot()

    # ggplot for simple linear regression
    df_ggplot = pd.DataFrame({'Feature 1': X.flatten(), 'Target': y})
    p = (ggplot(df_ggplot, aes(x='Feature 1', y='Target')) +
         geom_point(color='blue', alpha=0.6, size=2) +
         geom_line(aes(y=model.predict(X)), color='red', size=1.5) +
         geom_hline(yintercept=np.mean(y), linetype='dashed', color='orange') +
         labs(title='Simple Linear Regression (ggplot)', 
              x='Feature 1', 
              y='Target Variable') +
         theme(subplots_adjust={'top': 0.9}))

    st.pyplot(p.draw())

def plot_3d_regression(X, y, model, X_test, y_test, y_pred):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data Points', alpha=0.6)

    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    y_grid = model.intercept_ + model.coef_[0] * x1_grid + model.coef_[1] * x2_grid

    ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.5)

    mean_y = np.mean(y)
    mean_plane_height = mean_y * np.ones_like(x1_grid)
    ax.plot_surface(x1_grid, x2_grid, mean_plane_height, color='yellow', alpha=0.5)

    for x_val, y_actual, y_predicted in zip(X_test, y_test, y_pred):
        ax.plot([x_val[0], x_val[0]], [x_val[1], x_val[1]], [y_predicted, y_actual], color='green', linestyle=':', linewidth=2)
        ax.plot([x_val[0], x_val[0]], [x_val[1], x_val[1]], [mean_y, y_predicted], color='purple', linestyle='--', linewidth=2)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target Variable')
    ax.set_title('3D Multiple Linear Regression')
    ax.view_init(elev=20, azim=30)
    st.pyplot(fig)

    # ggplot for 2D projection of features
    df_ggplot = pd.DataFrame({'Feature 1': X[:, 0], 'Feature 2': X[:, 1], 'Target': y})
    p2 = (ggplot(df_ggplot, aes(x='Feature 1', y='Feature 2', color='Target')) +
          geom_point() +
          labs(title='2D Scatter Plot of Features (ggplot)', 
               x='Feature 1', 
               y='Feature 2') +
          theme(subplots_adjust={'top': 0.9}))

    st.pyplot(p2.draw())

def plot_pair_plot(df):
    st.subheader("Pair Plots")
    pair_plot_data = df.sample(min(500, len(df)))  # Sample for better performance if data is large
    sns.pairplot(pair_plot_data, diag_kind='kde', plot_kws={'alpha': 0.5})
    st.pyplot()

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, color='green')
    plt.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=2)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid()
    st.pyplot()

def plot_residuals(y_pred, residuals):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid()
    st.pyplot()

def plot_histogram_of_residuals(residuals):
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid()
    st.pyplot()