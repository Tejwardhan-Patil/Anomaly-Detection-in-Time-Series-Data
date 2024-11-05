import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_date_format(data, date_column):
    """
    Validates the date format in the time-series data.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data.
        date_column (str): Name of the column containing the date.

    Raises:
        ValueError: If the date format is invalid or not recognized.
    """
    try:
        pd.to_datetime(data[date_column])
        logging.info("Date format validation passed.")
    except Exception as e:
        logging.error(f"Invalid date format: {e}")
        raise ValueError("Invalid date format detected in the date column.")

def check_missing_timestamps(data, date_column):
    """
    Checks for missing timestamps in the time-series data.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data.
        date_column (str): Name of the column containing the date.

    Returns:
        bool: True if there are missing timestamps, False otherwise.
    """
    data[date_column] = pd.to_datetime(data[date_column])
    if data[date_column].isnull().any():
        logging.error("Missing timestamps found.")
        raise ValueError("There are missing timestamps in the time-series data.")
    logging.info("No missing timestamps found.")

def plot_splits(train_data, validation_data, test_data, date_column):
    """
    Plots the splits of train, validation, and test data based on time.

    Args:
        train_data (pd.DataFrame): Training set.
        validation_data (pd.DataFrame): Validation set.
        test_data (pd.DataFrame): Test set.
        date_column (str): Name of the column containing the date.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_data[date_column], train_data.index, label='Training Data', color='blue')
    plt.plot(validation_data[date_column], validation_data.index, label='Validation Data', color='orange')
    plt.plot(test_data[date_column], test_data.index, label='Test Data', color='green')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Index')
    plt.title('Train/Validation/Test Split')
    plt.grid(True)
    plt.show()
    logging.info("Data splits plotted successfully.")

def split_time_series(data, train_size=0.7, validation_size=0.15, test_size=0.15, date_column='date', visualize=False):
    """
    Splits the time-series data into train, validation, and test sets based on the time sequence.

    Args:
        data (pd.DataFrame): Time-series data as a pandas DataFrame.
        train_size (float): Proportion of the data to include in the training set.
        validation_size (float): Proportion of the data to include in the validation set.
        test_size (float): Proportion of the data to include in the test set.
        date_column (str): Name of the column containing the date or timestamp.
        visualize (bool): Whether to visualize the data splits.

    Returns:
        tuple: DataFrames for train, validation, and test sets.
    """
    logging.info("Starting time-series data split.")
    
    # Validate date format
    validate_date_format(data, date_column)
    
    # Sort data by the date column
    data = data.sort_values(by=date_column)
    logging.info("Data sorted by date.")
    
    # Check for missing timestamps
    check_missing_timestamps(data, date_column)
    
    # Calculate the split points
    train_end = int(len(data) * train_size)
    validation_end = train_end + int(len(data) * validation_size)
    
    # Split the data
    train_data = data.iloc[:train_end]
    validation_data = data.iloc[train_end:validation_end]
    test_data = data.iloc[validation_end:]
    
    logging.info(f"Data split into training ({len(train_data)}), validation ({len(validation_data)}), and test ({len(test_data)}) sets.")
    
    if visualize:
        plot_splits(train_data, validation_data, test_data, date_column)
    
    return train_data, validation_data, test_data

def save_splits(train_data, validation_data, test_data, output_dir):
    """
    Saves the split data sets to CSV files.

    Args:
        train_data (pd.DataFrame): Training set.
        validation_data (pd.DataFrame): Validation set.
        test_data (pd.DataFrame): Test set.
        output_dir (str): Directory to save the output files.
    """
    train_path = f'{output_dir}/train_data.csv'
    validation_path = f'{output_dir}/validation_data.csv'
    test_path = f'{output_dir}/test_data.csv'
    
    train_data.to_csv(train_path, index=False)
    validation_data.to_csv(validation_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    logging.info(f"Training data saved to {train_path}")
    logging.info(f"Validation data saved to {validation_path}")
    logging.info(f"Test data saved to {test_path}")

def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Split time-series data into train, validation, and test sets.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing the time-series data.')
    parser.add_argument('output_dir', type=str, help='Directory where the output CSV files will be saved.')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion of data for training set.')
    parser.add_argument('--validation_size', type=float, default=0.15, help='Proportion of data for validation set.')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion of data for test set.')
    parser.add_argument('--date_column', type=str, default='date', help='Column containing the date or timestamp.')
    parser.add_argument('--visualize', action='store_true', help='Visualize the train/validation/test split.')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_arguments()
    
    # Load the data
    logging.info(f"Loading data from {args.input_file}")
    data = pd.read_csv(args.input_file)
    
    # Perform the split
    train_data, validation_data, test_data = split_time_series(
        data, 
        train_size=args.train_size, 
        validation_size=args.validation_size, 
        test_size=args.test_size, 
        date_column=args.date_column, 
        visualize=args.visualize
    )
    
    # Save the splits
    save_splits(train_data, validation_data, test_data, args.output_dir)
    
    logging.info("Data splitting process completed.")