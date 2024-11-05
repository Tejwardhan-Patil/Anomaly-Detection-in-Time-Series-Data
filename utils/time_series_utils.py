import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.fft import fft, ifft
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

def rolling_window(data, window_size, step_size=1):
    """
    Creates a rolling window over the time-series data.
    
    Parameters:
    data (pd.Series or np.array): The time-series data to apply the rolling window to.
    window_size (int): The size of the window.
    step_size (int): The step size for moving the window.

    Returns:
    np.array: Array of windows created from the data.
    """
    num_windows = (len(data) - window_size) // step_size + 1
    return np.array([data[i:i+window_size] for i in range(0, num_windows * step_size, step_size)])

def remove_trend(data, method='linear'):
    """
    Removes trend from time-series data using different methods.
    
    Parameters:
    data (pd.Series or np.array): The time-series data to detrend.
    method (str): Method for detrending ('linear', 'constant', 'moving_average').

    Returns:
    np.array: Detrended data.
    """
    if method == 'linear':
        return detrend(data, type='linear')
    elif method == 'constant':
        return detrend(data, type='constant')
    elif method == 'moving_average':
        return data - calculate_moving_average(data, window_size=10)
    else:
        raise ValueError("Unsupported method for detrending")

def normalize_series(data, method='zscore'):
    """
    Normalizes time-series data to have mean 0 and standard deviation 1, or scales to a range.
    
    Parameters:
    data (pd.Series or np.array): The time-series data to normalize.
    method (str): Method for normalization ('zscore' or 'minmax').

    Returns:
    np.array: Normalized data.
    """
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Unsupported method for normalization")

def resample_data(data, frequency):
    """
    Resamples time-series data to a new frequency.
    
    Parameters:
    data (pd.Series): The time-series data to resample.
    frequency (str): The new frequency (e.g., 'D' for daily, 'H' for hourly).

    Returns:
    pd.Series: Resampled data.
    """
    return data.resample(frequency).mean()

def calculate_moving_average(data, window_size):
    """
    Calculates moving average over the time-series data.
    
    Parameters:
    data (pd.Series or np.array): The time-series data to apply moving average to.
    window_size (int): The size of the moving window.

    Returns:
    pd.Series or np.array: Moving average of the data.
    """
    return data.rolling(window=window_size).mean()

def compute_lag_features(data, lags):
    """
    Creates lag features for the time-series data.
    
    Parameters:
    data (pd.Series): The time-series data to create lag features from.
    lags (list of int): List of lag steps to generate features for.

    Returns:
    pd.DataFrame: DataFrame containing the lag features.
    """
    return pd.concat([data.shift(lag) for lag in lags], axis=1)

def seasonal_decompose(data, period):
    """
    Decomposes the time-series into trend, seasonal, and residual components.
    
    Parameters:
    data (pd.Series): The time-series data to decompose.
    period (int): The period for seasonal decomposition.

    Returns:
    DecomposeResult: A result object with trend, seasonal, and residual components.
    """
    return seasonal_decompose(data, period=period)

def fourier_transform(data):
    """
    Applies Fourier Transform to the time-series data to extract frequency components.
    
    Parameters:
    data (pd.Series or np.array): The time-series data to apply Fourier Transform to.

    Returns:
    np.array: Transformed data in the frequency domain.
    """
    return fft(data)

def inverse_fourier_transform(data):
    """
    Applies Inverse Fourier Transform to the frequency domain data to get back time-domain data.
    
    Parameters:
    data (np.array): The frequency domain data.

    Returns:
    np.array: Transformed data in the time domain.
    """
    return ifft(data)

def detect_outliers(data, z_thresh=3):
    """
    Detects outliers in the time-series data based on Z-score thresholding.
    
    Parameters:
    data (pd.Series or np.array): The time-series data.
    z_thresh (float): The Z-score threshold for outlier detection.

    Returns:
    np.array: Boolean array where True represents an outlier.
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > z_thresh

def scale_data(data, method='standard'):
    """
    Scales time-series data using different methods (StandardScaler or MinMaxScaler).
    
    Parameters:
    data (pd.Series or np.array): The time-series data to scale.
    method (str): Method for scaling ('standard' or 'minmax').

    Returns:
    np.array: Scaled data.
    """
    if method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Unsupported scaling method")

def rolling_std(data, window_size):
    """
    Calculates the rolling standard deviation over a window.
    
    Parameters:
    data (pd.Series or np.array): The time-series data.
    window_size (int): The window size for calculating rolling standard deviation.

    Returns:
    pd.Series or np.array: Rolling standard deviation of the data.
    """
    return data.rolling(window=window_size).std()

def create_time_based_features(data, time_column):
    """
    Creates additional time-based features (hour, day of week, month) from time-series data.
    
    Parameters:
    data (pd.DataFrame): The data containing a time column.
    time_column (str): The column name for the time data.

    Returns:
    pd.DataFrame: DataFrame with additional time-based features.
    """
    data['hour'] = data[time_column].dt.hour
    data['day_of_week'] = data[time_column].dt.dayofweek
    data['month'] = data[time_column].dt.month
    return data

def apply_differencing(data, order=1):
    """
    Applies differencing to the time-series data to remove trends and make the data stationary.
    
    Parameters:
    data (pd.Series or np.array): The time-series data.
    order (int): The order of differencing to apply.

    Returns:
    pd.Series: Differenced time-series data.
    """
    return data.diff(periods=order).dropna()

def extract_rolling_features(data, window_size):
    """
    Extracts rolling features such as mean, standard deviation, and min/max.
    
    Parameters:
    data (pd.Series): The time-series data.
    window_size (int): The window size for calculating rolling statistics.

    Returns:
    pd.DataFrame: DataFrame containing rolling statistics.
    """
    features = pd.DataFrame()
    features['rolling_mean'] = data.rolling(window=window_size).mean()
    features['rolling_std'] = data.rolling(window=window_size).std()
    features['rolling_min'] = data.rolling(window=window_size).min()
    features['rolling_max'] = data.rolling(window=window_size).max()
    return features

def exponential_moving_average(data, span):
    """
    Calculates the exponential moving average for time-series data.
    
    Parameters:
    data (pd.Series): The time-series data.
    span (int): The span for calculating the exponential moving average.

    Returns:
    pd.Series: Exponential moving average of the data.
    """
    return data.ewm(span=span).mean()

def remove_seasonality(data, period):
    """
    Removes seasonality from time-series data by subtracting the seasonal component.
    
    Parameters:
    data (pd.Series): The time-series data.
    period (int): The period for identifying the seasonality.

    Returns:
    pd.Series: Time-series data with seasonality removed.
    """
    decomposition = seasonal_decompose(data, period=period)
    return data - decomposition.seasonal