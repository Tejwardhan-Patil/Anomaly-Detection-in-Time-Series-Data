import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model
from joblib import load
from utils.data_loader import load_data
from utils.metrics import calculate_anomaly_scores
from utils.visualization import plot_anomalies

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration paths for models
MODEL_PATHS = {
    'lstm': 'models/deep_learning/lstm_model.h5',
    'autoencoder': 'models/deep_learning/autoencoder_model.h5',
    'random_forest': 'models/traditional/random_forest_model.joblib',
    'isolation_forest': 'models/unsupervised/isolation_forest_model.joblib'
}

def load_model_by_type(model_type):
    """Load the appropriate model based on the model type."""
    logger.info(f"Loading {model_type} model.")
    if model_type == 'lstm' or model_type == 'autoencoder':
        try:
            model = load_model(MODEL_PATHS[model_type])
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            raise
    else:
        try:
            model = load(MODEL_PATHS[model_type])
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            raise
    logger.info(f"{model_type} model loaded successfully.")
    return model

def preprocess_data(data):
    """Preprocess time-series data before feeding it to the models."""
    logger.info("Preprocessing data.")
    # Handle missing values, normalization, or other preprocessing techniques
    try:
        # Normalization
        data = (data - np.mean(data)) / np.std(data)
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
    logger.info("Data preprocessing completed.")
    return data

def detect_anomalies(data, model, model_type):
    """Run inference on the time-series data using the chosen model."""
    logger.info(f"Running anomaly detection using {model_type}.")
    try:
        if model_type in ['lstm', 'autoencoder']:
            reconstructed_data = model.predict(data)
            anomaly_scores = np.mean(np.abs(reconstructed_data - data), axis=1)
        elif model_type == 'random_forest':
            anomaly_scores = model.predict_proba(data)[:, 1]  # Probability of being an anomaly
        elif model_type == 'isolation_forest':
            anomaly_scores = -model.decision_function(data)
        else:
            raise ValueError(f"Model type {model_type} is not supported.")
    except Exception as e:
        logger.error(f"Error during anomaly detection: {str(e)}")
        raise
    logger.info(f"Anomaly detection using {model_type} completed.")
    return anomaly_scores

def calculate_threshold(anomaly_scores, percentile=95):
    """Calculate anomaly threshold based on the anomaly scores."""
    logger.info(f"Calculating anomaly threshold at the {percentile}th percentile.")
    try:
        threshold = np.percentile(anomaly_scores, percentile)
    except Exception as e:
        logger.error(f"Error calculating threshold: {str(e)}")
        raise
    logger.info(f"Threshold set at: {threshold}")
    return threshold

def identify_anomalies(anomaly_scores, threshold):
    """Identify whether each data point is an anomaly based on the threshold."""
    logger.info("Identifying anomalies.")
    try:
        anomalies = anomaly_scores > threshold
    except Exception as e:
        logger.error(f"Error identifying anomalies: {str(e)}")
        raise
    logger.info(f"Anomalies identified: {np.sum(anomalies)} out of {len(anomalies)} points.")
    return anomalies

def visualize_anomalies(data, anomalies, model_type):
    """Visualize the anomalies on the time-series data."""
    logger.info(f"Visualizing anomalies detected by {model_type}.")
    try:
        plot_anomalies(data, anomalies, title=f"Anomalies detected by {model_type}")
        plt.show()
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        raise
    logger.info(f"Visualization completed.")

def save_results(data, anomalies, output_path='results/anomalies.csv'):
    """Save the anomaly detection results to a CSV file."""
    logger.info(f"Saving results to {output_path}.")
    try:
        results = pd.DataFrame(data)
        results['is_anomaly'] = anomalies
        results.to_csv(output_path, index=False)
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise
    logger.info("Results saved successfully.")

def handle_model_type(model_type):
    """Ensure model type is valid."""
    if model_type not in MODEL_PATHS.keys():
        logger.error(f"Invalid model type {model_type}. Available types: {list(MODEL_PATHS.keys())}")
        raise ValueError(f"Unsupported model type: {model_type}")
    logger.info(f"Model type {model_type} is valid.")

def run_inference(input_data_path, model_type, output_path='results/anomalies.csv'):
    """Main function to run inference on the time-series data."""
    # Validate model type
    handle_model_type(model_type)

    # Load the data
    logger.info(f"Loading data from {input_data_path}")
    try:
        data = load_data(input_data_path)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Preprocess the data
    data = preprocess_data(data)

    # Load the pre-trained model
    model = load_model_by_type(model_type)

    # Detect anomalies
    anomaly_scores = detect_anomalies(data, model, model_type)

    # Calculate the threshold for anomalies
    threshold = calculate_threshold(anomaly_scores)

    # Identify anomalies
    anomalies = identify_anomalies(anomaly_scores, threshold)

    # Visualize anomalies
    visualize_anomalies(data, anomalies, model_type)

    # Save the results to a CSV file
    save_results(data, anomalies, output_path)

if __name__ == '__main__':
    input_data_path = 'data/processed/new_data.csv'
    model_type = 'lstm'  # 'lstm', 'autoencoder', 'random_forest', 'isolation_forest'
    output_path = 'results/anomalies_lstm.csv'
    
    try:
        run_inference(input_data_path, model_type, output_path)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")