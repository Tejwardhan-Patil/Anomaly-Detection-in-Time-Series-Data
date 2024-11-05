import os
import pandas as pd
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_file(filepath):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    logging.info(f"Validated file: {filepath}")

def validate_directory(directory_path):
    if not os.path.exists(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    logging.info(f"Validated directory: {directory_path}")

def load_csv(filepath, parse_dates=True, index_col=None, **kwargs):
    validate_file(filepath)
    logging.info(f"Loading CSV file: {filepath}")
    
    return pd.read_csv(filepath, parse_dates=parse_dates, index_col=index_col, **kwargs)

def load_json(filepath):
    validate_file(filepath)
    logging.info(f"Loading JSON file: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def load_excel(filepath, sheet_name=0, **kwargs):
    validate_file(filepath)
    logging.info(f"Loading Excel file: {filepath}")
    
    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

def load_parquet(filepath, **kwargs):
    validate_file(filepath)
    logging.info(f"Loading Parquet file: {filepath}")
    
    return pd.read_parquet(filepath, **kwargs)

def filter_files(directory_path, file_extension=None):
    validate_directory(directory_path)
    
    filtered_files = []
    for file in os.listdir(directory_path):
        if file_extension and file.endswith(file_extension):
            filtered_files.append(file)
        elif not file_extension:
            filtered_files.append(file)
    
    logging.info(f"Filtered files: {filtered_files}")
    return filtered_files

def load_from_directory(directory_path, file_format='csv'):
    validate_directory(directory_path)
    
    data = {}
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        
        if file_format == 'csv' and file.endswith('.csv'):
            data[file] = load_csv(file_path)
        elif file_format == 'json' and file.endswith('.json'):
            data[file] = load_json(file_path)
        elif file_format == 'excel' and file.endswith('.xlsx'):
            data[file] = load_excel(file_path)
        elif file_format == 'parquet' and file.endswith('.parquet'):
            data[file] = load_parquet(file_path)
    
    logging.info(f"Loaded {len(data)} files from directory: {directory_path}")
    return data

def load_data(filepath, file_format='csv', **kwargs):
    if file_format == 'csv':
        return load_csv(filepath, **kwargs)
    elif file_format == 'json':
        return load_json(filepath)
    elif file_format == 'excel':
        return load_excel(filepath, **kwargs)
    elif file_format == 'parquet':
        return load_parquet(filepath, **kwargs)
    else:
        logging.error(f"Unsupported file format: {file_format}")
        raise ValueError(f"Unsupported file format: {file_format}")

def load_multiple_files(filepaths, file_format='csv'):
    data = {}
    for filepath in filepaths:
        logging.info(f"Loading file: {filepath}")
        data[os.path.basename(filepath)] = load_data(filepath, file_format=file_format)
    return data

def preprocess_data(df, dropna=True, fillna_value=None, normalize=False):
    if dropna:
        logging.info("Dropping missing values")
        df = df.dropna()
    
    if fillna_value is not None:
        logging.info(f"Filling missing values with {fillna_value}")
        df = df.fillna(fillna_value)
    
    if normalize:
        logging.info("Normalizing data")
        df = (df - df.mean()) / df.std()

    return df

def clean_data(df, columns=None, rename_columns=None):
    if columns:
        logging.info(f"Selecting columns: {columns}")
        df = df[columns]
    
    if rename_columns:
        logging.info(f"Renaming columns: {rename_columns}")
        df = df.rename(columns=rename_columns)
    
    return df

def split_data_by_time(df, time_column, train_ratio=0.8):
    df = df.sort_values(by=time_column)
    split_point = int(len(df) * train_ratio)
    train_data = df.iloc[:split_point]
    test_data = df.iloc[split_point:]
    
    logging.info(f"Split data into train ({len(train_data)}) and test ({len(test_data)})")
    return train_data, test_data

def save_csv(df, filepath):
    logging.info(f"Saving DataFrame to CSV: {filepath}")
    df.to_csv(filepath, index=False)

def save_json(data, filepath):
    logging.info(f"Saving data to JSON: {filepath}")
    with open(filepath, 'w') as f:
        json.dump(data, f)

def save_parquet(df, filepath):
    logging.info(f"Saving DataFrame to Parquet: {filepath}")
    df.to_parquet(filepath)

def archive_files(directory_path, archive_name):
    import shutil
    validate_directory(directory_path)
    
    logging.info(f"Archiving directory: {directory_path} into {archive_name}.zip")
    shutil.make_archive(archive_name, 'zip', directory_path)

def filter_files_by_date(directory_path, date_column, start_date=None, end_date=None):
    validate_directory(directory_path)
    
    filtered_files = []
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        file_stat = os.stat(file_path)
        file_date = datetime.fromtimestamp(file_stat.st_mtime)
        
        if start_date and end_date:
            if start_date <= file_date <= end_date:
                filtered_files.append(file)
        elif start_date and file_date >= start_date:
            filtered_files.append(file)
        elif end_date and file_date <= end_date:
            filtered_files.append(file)
        else:
            filtered_files.append(file)
    
    logging.info(f"Filtered files by date: {filtered_files}")
    return filtered_files

def convert_to_datetime(df, columns):
    for col in columns:
        logging.info(f"Converting column to datetime: {col}")
        df[col] = pd.to_datetime(df[col])
    
    return df

def aggregate_time_series(df, time_column, freq='D', agg_func='mean'):
    logging.info(f"Aggregating time-series data by {freq} with {agg_func}")
    df[time_column] = pd.to_datetime(df[time_column])
    return df.set_index(time_column).resample(freq).agg(agg_func)

def check_missing_values(df):
    missing = df.isnull().sum()
    if missing.any():
        logging.info(f"Missing values detected: {missing}")
    else:
        logging.info("No missing values detected")
    return missing

def describe_data(df):
    logging.info("Generating summary statistics for the dataset")
    return df.describe()

def remove_outliers(df, columns, threshold=1.5):
    logging.info("Removing outliers using the IQR method")
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[columns] < (Q1 - threshold * IQR)) | (df[columns] > (Q3 + threshold * IQR))).any(axis=1)]