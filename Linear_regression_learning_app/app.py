# app.py
import streamlit as st
import pandas as pd
from random_data_generator import generate_random_sample
from regression_model import fit_model
from visualization import (
    plot_simple_regression,
    plot_3d_regression,
    plot_pair_plot,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_histogram_of_residuals,
)

# Set the page config to wide
st.set_page_config(page_title="Linear Regression Learning App", layout="wide")

# Title of the app
st.title("📊 Linear Regression Learning App")

# Sidebar for user input
st.sidebar.header("User Input")
sample_size = st.sidebar.slider("Select Sample Size", min_value=1, max_value=500, value=100)
num_features = st.sidebar.slider("Number of Features", min_value=1, max_value=5, value=2)
include_outliers = st.sidebar.checkbox("Include Outliers", value=False)

# Generate random data
X, y = generate_random_sample(sample_size, num_features, include_outliers)

# Fit the regression model
model, X_test, y_test = fit_model(X, y)

# Predictions
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Create DataFrame for visualization
df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(num_features)])
df['Target'] = y

# Visualization
if num_features == 1:
    plot_simple_regression(X, y, model, X_test, y_test, y_pred)
elif num_features == 2:
    plot_3d_regression(X, y, model, X_test, y_test, y_pred)
else:
    plot_pair_plot(df)

# Regression Coefficients and Statistical Measures
st.subheader("Regression Coefficients")
st.write(f"Intercept: {model.intercept_:.2f}")
for i, coef in enumerate(model.coef_):
    st.write(f"Slope (Feature {i+1}): {coef:.2f}")

# Actual vs Predicted Plot
plot_actual_vs_predicted(y_test, y_pred)

# Residuals Plot
plot_residuals(y_pred, residuals)

# Histogram of Residuals
plot_histogram_of_residuals(residuals)

# Option to download the dataset
st.sidebar.subheader("Download Generated Data")
csv = df.to_csv(index=False)
st.sidebar.download_button("Download CSV", csv, "generated_data.csv", "text/csv")
