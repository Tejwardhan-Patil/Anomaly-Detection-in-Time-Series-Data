import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import pandas as pd

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_data(filepath):
    """
    Load time-series data from a CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file containing time-series data.
    
    Returns:
        pandas.DataFrame: Time-series data as a DataFrame.
    """
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df, columns_to_scale):
    """
    Preprocess the time-series data by scaling specified columns.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing time-series data.
        columns_to_scale (list): List of column names to scale.
    
    Returns:
        numpy.ndarray: Scaled data as a numpy array.
    """
    logger.info("Preprocessing data (scaling features)")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns_to_scale])
    return scaled_data

def dbscan_anomaly_detection(data, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering for anomaly detection on time-series data.
    
    Parameters:
        data (numpy.ndarray): Scaled time-series data.
        eps (float): The maximum distance between two samples for one to be considered in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood to form a cluster.
    
    Returns:
        numpy.ndarray: Array where anomalies are marked with -1 and clusters are labeled with integers.
    """
    logger.info(f"Running DBSCAN with eps={eps} and min_samples={min_samples}")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels

def identify_anomalies(labels):
    """
    Identify anomalies based on DBSCAN labels.
    
    Parameters:
        labels (numpy.ndarray): Labels from DBSCAN clustering.
    
    Returns:
        numpy.ndarray: Indices of detected anomalies.
    """
    logger.info("Identifying anomalies")
    return np.where(labels == -1)[0]

def visualize_clusters(data, labels, anomalies):
    """
    Visualize the clustering results and mark anomalies.
    
    Parameters:
        data (numpy.ndarray): Time-series data as a 2D numpy array.
        labels (numpy.ndarray): Labels from DBSCAN clustering.
        anomalies (numpy.ndarray): Indices of detected anomalies.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.scatter(data[anomalies, 0], data[anomalies, 1], color='red', marker='x', s=100, label='Anomalies')
    plt.title('DBSCAN Clustering and Anomalies')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def save_anomalies(anomalies, df, output_filepath):
    """
    Save the detected anomalies to a CSV file.
    
    Parameters:
        anomalies (numpy.ndarray): Indices of detected anomalies.
        df (pandas.DataFrame): Original DataFrame containing time-series data.
        output_filepath (str): Path to save the anomalies CSV.
    """
    logger.info(f"Saving anomalies to {output_filepath}")
    anomaly_data = df.iloc[anomalies]
    anomaly_data.to_csv(output_filepath, index=False)

def log_results(labels):
    """
    Log clustering results such as number of clusters and anomalies.
    
    Parameters:
        labels (numpy.ndarray): Labels from DBSCAN clustering.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_anomalies = list(labels).count(-1)
    logger.info(f"Detected {n_clusters} clusters")
    logger.info(f"Detected {n_anomalies} anomalies")

def run_dbscan_pipeline(filepath, output_filepath, columns_to_scale, eps=0.5, min_samples=5):
    """
    Full pipeline to run DBSCAN anomaly detection on a time-series dataset.
    
    Parameters:
        filepath (str): Path to the input CSV file.
        output_filepath (str): Path to save the anomalies CSV.
        columns_to_scale (list): List of columns to scale.
        eps (float): The maximum distance between two samples for DBSCAN.
        min_samples (int): Minimum samples to form a cluster in DBSCAN.
    """
    # Load and preprocess data
    df = load_data(filepath)
    data = preprocess_data(df, columns_to_scale)
    
    # Run DBSCAN and identify anomalies
    labels = dbscan_anomaly_detection(data, eps=eps, min_samples=min_samples)
    anomalies = identify_anomalies(labels)
    
    # Visualize clusters and anomalies
    visualize_clusters(data, labels, anomalies)
    
    # Log results and save anomalies
    log_results(labels)
    save_anomalies(anomalies, df, output_filepath)

if __name__ == "__main__":
    # Configuration for DBSCAN pipeline
    config = {
        'input_filepath': 'data/time_series_data.csv',
        'output_filepath': 'data/anomalies.csv',
        'columns_to_scale': ['feature1', 'feature2'],
        'eps': 0.3,
        'min_samples': 10
    }
    
    # Run the DBSCAN anomaly detection pipeline
    run_dbscan_pipeline(
        filepath=config['input_filepath'],
        output_filepath=config['output_filepath'],
        columns_to_scale=config['columns_to_scale'],
        eps=config['eps'],
        min_samples=config['min_samples']
    )