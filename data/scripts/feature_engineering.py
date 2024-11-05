import numpy as np
import pandas as pd
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose

# Load time-series data
def load_data(file_path):
    """Load time-series data from a CSV file"""
    try:
        data = pd.read_csv(file_path, parse_dates=True, index_col=0)
        return data
    except Exception as e:
        raise IOError(f"Error loading data: {e}")

# Generate rolling statistical features (mean, std, min, max, skewness, kurtosis)
def generate_rolling_features(data, window_size=5):
    """Generate rolling statistical features"""
    rolling_data = data.rolling(window=window_size)
    rolling_features = pd.DataFrame()

    rolling_features['rolling_mean'] = rolling_data.mean()
    rolling_features['rolling_std'] = rolling_data.std()
    rolling_features['rolling_min'] = rolling_data.min()
    rolling_features['rolling_max'] = rolling_data.max()
    rolling_features['rolling_skew'] = rolling_data.skew()
    rolling_features['rolling_kurt'] = rolling_data.kurt()

    return rolling_features

# Generate lag features for a range of lags
def generate_lag_features(data, max_lag=5):
    """Generate lag features for the data"""
    lag_features = pd.DataFrame()
    for lag in range(1, max_lag + 1):
        lag_features[f'lag_{lag}'] = data.shift(lag)
    return lag_features

# Generate difference features
def generate_difference_features(data, periods=1):
    """Generate difference features (first or second-order differences)"""
    difference_features = pd.DataFrame()
    difference_features[f'diff_{periods}'] = data.diff(periods).fillna(0)
    return difference_features

# Generate cumulative features (cumulative sum)
def generate_cumulative_features(data):
    """Generate cumulative sum and product features"""
    cumulative_features = pd.DataFrame()
    cumulative_features['cumulative_sum'] = data.cumsum()
    cumulative_features['cumulative_prod'] = data.cumprod()
    return cumulative_features

# Generate Fourier Transform features
def generate_fft_features(data, n_components=10):
    """Generate features based on the Fourier Transform"""
    fft_values = fft(data, axis=0)
    fft_features = pd.DataFrame()

    for i in range(n_components):
        fft_features[f'fft_{i}_real'] = np.real(fft_values[:, i])
        fft_features[f'fft_{i}_imag'] = np.imag(fft_values[:, i])

    return fft_features

# Decompose time series into trend, seasonal, and residual components
def decompose_time_series(data, model='additive'):
    """Perform seasonal decomposition on the time-series data"""
    decomposition = seasonal_decompose(data, model=model, period=12)
    decomposed_features = pd.DataFrame()
    
    decomposed_features['trend'] = decomposition.trend
    decomposed_features['seasonal'] = decomposition.seasonal
    decomposed_features['residual'] = decomposition.resid

    return decomposed_features.fillna(0)

# Apply PCA to reduce dimensionality of the feature set
def apply_pca(features, n_components=5):
    """Reduce dimensionality using PCA"""
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    pca_df = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(n_components)])
    return pca_df

# Scale features using Standard Scaler
def scale_features(features):
    """Standardize the features"""
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    return scaled_features

# Handle missing values by forward filling
def handle_missing_values(data):
    """Fill missing values in the time-series data"""
    return data.fillna(method='ffill')

# Generate statistical summary features
def generate_statistical_summary(data):
    """Generate a set of statistical summary features (mean, std, etc.)"""
    summary_features = pd.DataFrame()
    summary_features['mean'] = data.mean()
    summary_features['std'] = data.std()
    summary_features['min'] = data.min()
    summary_features['max'] = data.max()
    summary_features['median'] = data.median()
    summary_features['variance'] = data.var()
    return summary_features

# Save the generated features to a CSV file
def save_features(features, output_path):
    """Save the feature set to a CSV file"""
    try:
        features.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"Error saving features: {e}")

# Main feature engineering function
def generate_features(file_path, window_size=5, max_lag=5, periods=1, n_fft_components=10, n_pca_components=5):
    """Main function to generate all features from the data"""
    # Load raw data
    data = load_data(file_path)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Generate features
    rolling_features = generate_rolling_features(data, window_size)
    lag_features = generate_lag_features(data, max_lag)
    difference_features = generate_difference_features(data, periods)
    cumulative_features = generate_cumulative_features(data)
    fft_features = generate_fft_features(data, n_fft_components)
    decomposed_features = decompose_time_series(data)
    
    # Combine all features into one DataFrame
    combined_features = pd.concat([rolling_features, lag_features, difference_features, cumulative_features, fft_features, decomposed_features], axis=1)
    
    # Apply PCA for dimensionality reduction
    pca_features = apply_pca(combined_features, n_pca_components)
    
    # Add PCA features to the combined feature set
    combined_features = pd.concat([combined_features, pca_features], axis=1)

    # Scale the features
    scaled_features = scale_features(combined_features)
    
    return scaled_features

# Usage of the feature engineering pipeline
if __name__ == "__main__":
    input_file = 'data/processed/time_series.csv'
    output_file = 'data/features/engineered_features.csv'

    # Generate features from the raw data
    features = generate_features(input_file)

    # Save the engineered features to a CSV file
    save_features(features, output_file)