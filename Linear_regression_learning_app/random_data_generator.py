# random_data_generator.py
import numpy as np

def generate_random_sample(sample_size, num_features, include_outliers=False):
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
