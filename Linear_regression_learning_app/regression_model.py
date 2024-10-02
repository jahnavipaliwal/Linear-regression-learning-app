# regression_model.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def fit_model(X, y):
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y  # Use the same data for testing

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    X_with_intercept = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
    hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T

    # Calculate beta cap (coefficients)
    beta_cap = model.coef_

    return model, X_test, y_test, hat_matrix, beta_cap
