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
    plot_hat_matrix,
    plot_beta_cap,
    plot_vif
)

# Set the page config to wide
st.set_page_config(page_title="Linear Regression Learning App", layout="wide")

# Title of the app
st.title("📊 Linear Regression Learning App")

# Sidebar for user input
st.sidebar.header("User Input")
sample_size = st.sidebar.slider("Select Sample Size", min_value=1, max_value=2000, value=10)
num_features = st.sidebar.slider("Number of Features", min_value=1, max_value=5, value=2)
include_outliers = st.sidebar.checkbox("Include Outliers", value=False)
include_multicollinearity = st.sidebar.checkbox("Introduce Multicollinearity", value=False)

# Store parameters in session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

if (
    st.session_state.data_generated is False or
    st.session_state.sample_size != sample_size or
    st.session_state.num_features != num_features or
    st.session_state.include_outliers != include_outliers or
    st.session_state.include_multicollinearity != include_multicollinearity
):
    # Generate random data
    X, y = generate_random_sample(sample_size, num_features, include_outliers, include_multicollinearity)
    st.session_state.data_generated = True
    st.session_state.sample_size = sample_size
    st.session_state.num_features = num_features
    st.session_state.include_outliers = include_outliers
    st.session_state.include_multicollinearity = include_multicollinearity

    # Fit the regression model
    model, X_test, y_test, hat_matrix, beta_cap = fit_model(X, y)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Create DataFrame for visualization
    df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(num_features)])
    df['Target'] = y

    # Store data and model in session state
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.hat_matrix = hat_matrix
    st.session_state.beta_cap = beta_cap
    st.session_state.df = df
    st.session_state.residuals = residuals  # Store residuals in session state
    st.session_state.y_pred = y_pred
else:
    # Use existing data and model from session state
    X = st.session_state.X
    y = st.session_state.y
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    hat_matrix = st.session_state.hat_matrix
    beta_cap = st.session_state.beta_cap
    df = st.session_state.df
    residuals = st.session_state.residuals  # Retrieve residuals from session state
    y_pred = st.session_state.y_pred
# Visualization selection
if num_features == 1:
    visualizations = ["Regression Graph", "Actual vs Predicted", "Residual Plot", "Histogram of Residuals"]
elif num_features == 2:
    visualizations = ["3D Regression Graph", "Hat Matrix", "Beta Cap", "Actual vs Predicted", "Residual Plot", "Histogram of Residuals"]
else:  # num_features > 2
    visualizations = ["Pair Plot", "Hat Matrix", "Beta Cap", "Actual vs Predicted", "Residual Plot", "Histogram of Residuals"]

selected_viz = st.sidebar.selectbox("Select Visualization", visualizations)

# Show selected visualization
if selected_viz == "Regression Graph":
    plot_simple_regression(X, y, model, X_test, y_test)
elif selected_viz == "3D Regression Graph":
    plot_3d_regression(X, y, model)
elif selected_viz == "Pair Plot":
    plot_pair_plot(df)
elif selected_viz == "Hat Matrix":
    plot_hat_matrix(hat_matrix)
elif selected_viz == "Beta Cap":
    plot_beta_cap(beta_cap)
elif selected_viz == "Actual vs Predicted":
    plot_actual_vs_predicted(y_test, y_pred)
elif selected_viz == "Residual Plot":
    plot_residuals(y_pred, residuals)
elif selected_viz == "Histogram of Residuals":
    plot_histogram_of_residuals(residuals)

# Regression Coefficients and Statistical Measures
st.subheader("Regression Coefficients")
st.write("### Estimated Coefficients")
st.write(f"Intercept: {model.intercept_:.2f}")
for i, coef in enumerate(model.coef_):
    st.write(f"**Slope (Feature {i+1}):** {coef:.2f}")

# Option to download the dataset
st.sidebar.subheader("Download Generated Data")
csv = df.to_csv(index=False)
st.sidebar.download_button("Download CSV", csv, "generated_data.csv", "text/csv")
