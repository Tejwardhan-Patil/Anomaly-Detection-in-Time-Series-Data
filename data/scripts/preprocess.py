import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
LOGS_PATH = "logs/preprocessing.log"

# Ensure processed data directory exists
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# Ensure logs directory exists
if not os.path.exists(os.path.dirname(LOGS_PATH)):
    os.makedirs(os.path.dirname(LOGS_PATH))

# Function to log messages to a file
def setup_file_logger():
    file_handler = logging.FileHandler(LOGS_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

setup_file_logger()

# Load raw data
def load_data(file_name):
    try:
        file_path = os.path.join(RAW_DATA_PATH, file_name)
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        logging.error(f"File {file_name} not found in {RAW_DATA_PATH}")
        raise
    except Exception as e:
        logging.error(f"Error loading file {file_name}: {str(e)}")
        raise

# Handle missing values
def handle_missing_values(data, method='interpolate'):
    try:
        logging.info(f"Handling missing values using method: {method}")
        if method == 'interpolate':
            return data.interpolate(method='time')
        elif method == 'fill_zero':
            return data.fillna(0)
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        logging.error(f"Error handling missing values: {str(e)}")
        raise

# Resample data (to hourly or daily frequency)
def resample_data(data, frequency='D'):
    try:
        logging.info(f"Resampling data to frequency: {frequency}")
        return data.resample(frequency).mean()
    except Exception as e:
        logging.error(f"Error during resampling: {str(e)}")
        raise

# Normalize data using MinMaxScaler
def normalize_data_minmax(data):
    try:
        logging.info("Normalizing data using MinMaxScaler")
        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
        return scaled_data
    except Exception as e:
        logging.error(f"Error during MinMax normalization: {str(e)}")
        raise

# Normalize data using StandardScaler
def normalize_data_standard(data):
    try:
        logging.info("Normalizing data using StandardScaler")
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
        return scaled_data
    except Exception as e:
        logging.error(f"Error during Standard normalization: {str(e)}")
        raise

# Validate data (checking for NaN values, ensuring correct data types)
def validate_data(data):
    logging.info("Validating data for preprocessing")
    if data.isnull().values.any():
        logging.warning("Data contains NaN values")
    if not np.issubdtype(data.index.dtype, np.datetime64):
        logging.warning("Index is not datetime, which is required for time-series processing")
    return data

# Save processed data
def save_processed_data(data, file_name):
    try:
        processed_file_path = os.path.join(PROCESSED_DATA_PATH, file_name)
        data.to_csv(processed_file_path)
        logging.info(f"Processed data saved to {processed_file_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {str(e)}")
        raise

# Detect outliers based on Z-score method
def detect_outliers(data, z_threshold=3):
    logging.info("Detecting outliers using Z-score method")
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = (z_scores > z_threshold)
    logging.info(f"Outliers detected: {outliers.sum().sum()} out of {data.size}")
    return outliers

# Main preprocessing function
def preprocess_data(file_name, missing_method='interpolate', resample_freq='D', normalization_method='minmax'):
    try:
        # Load data
        data = load_data(file_name)
        
        # Validate data
        data = validate_data(data)

        # Handle missing values
        data = handle_missing_values(data, method=missing_method)

        # Resample data
        data = resample_data(data, frequency=resample_freq)

        # Normalize data
        if normalization_method == 'minmax':
            data = normalize_data_minmax(data)
        elif normalization_method == 'standard':
            data = normalize_data_standard(data)
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

        # Detect and log outliers
        outliers = detect_outliers(data)

        # Save processed data
        processed_file_name = file_name.replace("raw", "processed")
        save_processed_data(data, processed_file_name)

        logging.info(f"Preprocessing complete for {file_name}")
    
    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

# Generate summary statistics for data
def generate_summary_statistics(data):
    logging.info("Generating summary statistics for the dataset")
    try:
        summary = data.describe()
        logging.info(f"Summary statistics:\n{summary}")
        return summary
    except Exception as e:
        logging.error(f"Error generating summary statistics: {str(e)}")
        raise

# Handle data with seasonality by applying differencing
def difference_data(data, order=1):
    logging.info(f"Applying differencing with order: {order}")
    try:
        differenced_data = data.diff(periods=order).dropna()
        logging.info(f"Differenced data shape: {differenced_data.shape}")
        return differenced_data
    except Exception as e:
        logging.error(f"Error during differencing: {str(e)}")
        raise

# Usage
if __name__ == "__main__":
    try:
        # File to preprocess
        file_name = 'raw_time_series.csv'
        
        # Preprocess data
        preprocess_data(file_name, missing_method='interpolate', resample_freq='D', normalization_method='minmax')

        logging.info("Script completed successfully")
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")