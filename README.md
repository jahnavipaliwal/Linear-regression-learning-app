# Linear Regression Learning App

📊 **Linear Regression Learning App** is a Streamlit application designed to help users understand linear regression through interactive visualizations. Users can generate random data, fit regression models, and visualize results using both Matplotlib and ggplot.

## Features

- **Random Data Generation**: Generate datasets with a specified number of features and optional outliers.
- **Linear Regression**: Fit simple or multiple linear regression models.
- **Interactive Visualizations**:
  - Simple linear regression plots.
  - 3D visualizations for multiple linear regression.
  - Pair plots for feature relationships.
  - Actual vs. predicted values plots.
  - Residuals analysis.
  - Histogram of residuals.
- **Downloadable Data**: Users can download the generated dataset as a CSV file.

## Requirements

Make sure you have the following packages installed:

- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `plotnine`

You can install the required packages using pip:

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn plotnine
```
# Usage 

1. **Clone the Repository**: Open your terminal and run the following commands:

   ```bash
   git clone https://github.com/yourusername/linear-regression-learning-app.git
   cd linear-regression-learning-app
2. **Run the streamlit app**: In the terminal, execute:
   
   ```bash
   streamlit run app.py
   ```
3. **Install the Required Packages**: Make sure to install the necessary Python packages by running:

   ```bash
   pip install streamlit numpy pandas matplotlib seaborn scikit-learn plotnine
   ```
   
4. **Open in Browser**

    Once the app is running, open your web browser and navigate to `http://localhost:8501`. You will see the user interface where you can:
    
    - **Select the sample size** for data generation.
    - **Choose the number of features** (1 to 5).
    - **Decide whether to include outliers** in the generated data.
    - **Visualize the results interactively**.

5. **Visualize Results**

    After generating data and fitting the regression model, you will see:
    
    - **Scatter plots** for regression analysis.
    - **3D plots** for two features.
    - **Pair plots** for datasets with more than two features.
    - Metrics like **R-squared**, **SSE**, and **SSR**.

6. **Download Data**

    You can download the generated dataset as a CSV file from the sidebar.
