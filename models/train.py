import os
import yaml
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils.data_loader import load_data
from utils.metrics import custom_metric
from utils.visualization import plot_metrics
from models.traditional.arima import ARIMAAnomalyDetector
from models.deep_learning.lstm import LSTMAnomalyDetector
from models.unsupervised.isolation_forest import IsolationForestAnomalyDetector
from models.ensemble.ensemble_model import EnsembleModel
import logging

# Set up logging
logging.basicConfig(filename='train.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from YAML
def load_config(config_path):
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded successfully")
    return config

# Map model name to actual model implementation
def get_model(model_name, config):
    logging.info(f"Initializing model: {model_name}")
    if model_name == 'ARIMA':
        return ARIMAAnomalyDetector(config['ARIMA'])
    elif model_name == 'LSTM':
        return LSTMAnomalyDetector(config['LSTM'])
    elif model_name == 'IsolationForest':
        return IsolationForestAnomalyDetector(config['IsolationForest'])
    elif model_name == 'Ensemble':
        return EnsembleModel(config['Ensemble'])
    else:
        logging.error(f"Unsupported model type: {model_name}")
        raise ValueError(f"Unsupported model type: {model_name}")

# Handle missing values in the dataset
def handle_missing_values(data, strategy="mean"):
    logging.info(f"Handling missing values with strategy: {strategy}")
    if strategy == "mean":
        return data.fillna(data.mean())
    elif strategy == "median":
        return data.fillna(data.median())
    elif strategy == "drop":
        return data.dropna()
    else:
        logging.error(f"Unknown missing value strategy: {strategy}")
        raise ValueError(f"Unknown missing value strategy: {strategy}")

# Train and evaluate the model with various metrics
def train_and_evaluate(model, X_train, y_train, X_val, y_val, iteration, config):
    logging.info(f"Training model in iteration {iteration}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    accuracy = accuracy_score(y_val, predictions)
    custom = custom_metric(y_val, predictions)
    
    logging.info(f"Iteration {iteration} - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}, Custom: {custom}")
    
    # Save model performance for this fold
    results = {
        'iteration': iteration,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'custom_metric': custom
    }
    
    results_file = f"results_iteration_{iteration}.csv"
    pd.DataFrame([results]).to_csv(results_file, index=False)
    logging.info(f"Results saved to {results_file}")

    # Visualize results
    if config['visualize']:
        plot_metrics(y_val, predictions, title=f"Iteration {iteration} Performance")
    
    return model

# Time-series cross-validation
def time_series_cv(data, labels, config):
    logging.info(f"Starting time-series cross-validation with {config['cross_validation']['n_splits']} splits")
    tscv = TimeSeriesSplit(n_splits=config['cross_validation']['n_splits'])
    for i, (train_idx, val_idx) in enumerate(tscv.split(data)):
        logging.info(f"Processing fold {i+1}")
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        yield X_train, X_val, y_train, y_val

# Main training loop
def main(config_path, missing_value_strategy, log_file):
    # Set up logging to file
    if log_file:
        logging.getLogger().handlers = []  # Clear previous handlers
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = load_config(config_path)
    
    # Load and preprocess data
    data, labels = load_data(config['data']['path'])
    data = handle_missing_values(data, strategy=missing_value_strategy)
    
    # Initialize model
    model_name = config['model']['name']
    model = get_model(model_name, config)
    
    # Cross-validation training loop
    for iteration, (X_train, X_val, y_train, y_val) in enumerate(time_series_cv(data, labels, config), 1):
        model = train_and_evaluate(model, X_train, y_train, X_val, y_val, iteration, config)
    
    # Save the final trained model
    model_save_path = os.path.join(config['model']['save_dir'], f"{model_name}_model.pkl")
    joblib.dump(model, model_save_path)
    logging.info(f"Final model saved to {model_save_path}")
    print(f"Model training complete. Final model saved to {model_save_path}")

# Argument parsing for CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a time-series anomaly detection model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--missing_value_strategy', type=str, default='mean', 
                        help='Strategy for handling missing values: "mean", "median", or "drop".')
    parser.add_argument('--log_file', type=str, help='Log file to store detailed training logs.')
    
    args = parser.parse_args()
    
    # Execute the main function
    main(args.config, args.missing_value_strategy, args.log_file)