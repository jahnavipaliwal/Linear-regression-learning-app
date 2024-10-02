# random_data_generator.py
import numpy as np

def generate_random_sample(sample_size, num_features, include_outliers=False, introduce_multicollinearity=False):
    if introduce_multicollinearity and num_features > 1:
        # Create a base feature
        X_base = np.random.rand(sample_size, 1) * 100
        # Create correlated features
        X = np.hstack([X_base] + [X_base + np.random.randn(sample_size, 1) * 5 for _ in range(num_features - 1)])
    else:
        X = np.random.rand(sample_size, num_features) * 100  # Random values between 0 and 100
    
    coefficients = np.random.rand(num_features) * 10  # Random coefficients
    y = X @ coefficients + np.random.randn(sample_size) * 10  # Linear relation with noise
    
    if include_outliers:
        # Introduce outliers
        num_outliers = int(sample_size * 0.1)  # 10% outliers
        outlier_X = np.random.rand(num_outliers, num_features) * 100
        outlier_y = (np.random.rand(num_outliers) * 50) + 200  # Outliers significantly higher than normal
        X = np.vstack((X, outlier_X))
        y = np.concatenate((y, outlier_y))
    
    return X, y
