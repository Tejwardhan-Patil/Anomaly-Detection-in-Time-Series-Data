import os
import yaml
import argparse
import logging
import time
from models import traditional, deep_learning, unsupervised, ensemble
from utils import data_loader, metrics
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load the configuration file in YAML format.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration parameters.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Config file {config_path} does not exist")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info("Configuration successfully loaded.")
    return config

def validate_config(config):
    """
    Validate the contents of the configuration file.

    Args:
        config (dict): Configuration dictionary to validate.

    Raises:
        ValueError: If required configurations are missing.
    """
    required_keys = ['data', 'model', 'output']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration: {key}")
            raise ValueError(f"Configuration must include '{key}'")
    
    if 'path' not in config['data']:
        raise ValueError("Data path must be specified in the configuration")

    if 'name' not in config['model'] or 'type' not in config['model']:
        raise ValueError("Model configuration must include 'name' and 'type'")

    if 'path' not in config['output']:
        raise ValueError("Output path must be specified in the configuration")

    logger.info("Configuration validation completed.")

def initialize_model(config):
    """
    Initialize the model based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        Model: Anomaly detection model instance.
    """
    model = None
    model_type = config['model']['type']
    model_name = config['model']['name']
    
    logger.info(f"Initializing model: {model_name} of type {model_type}")
    
    if model_type == 'traditional':
        model = getattr(traditional, model_name)(**config['model']['params'])
    elif model_type == 'deep_learning':
        model = getattr(deep_learning, model_name)(**config['model']['params'])
    elif model_type == 'unsupervised':
        model = getattr(unsupervised, model_name)(**config['model']['params'])
    elif model_type == 'ensemble':
        model = getattr(ensemble, model_name)(**config['model']['params'])
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Model {model_name} initialized successfully.")
    return model

def split_data(data, test_size):
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): Time-series data to split.
        test_size (float): Fraction of the data to reserve for testing.

    Returns:
        tuple: Training and testing datasets.
    """
    try:
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
        logger.info(f"Data split into train and test sets. Test size: {test_size}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error during data splitting: {str(e)}")
        raise e

def train_model(model, train_data):
    """
    Train the model on the training data.

    Args:
        model (Model): Anomaly detection model instance.
        train_data (pd.DataFrame): Data to train the model on.
    """
    logger.info(f"Starting model training with {len(train_data)} samples.")
    
    try:
        start_time = time.time()
        model.fit(train_data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Model training completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e

def run_inference(model, test_data):
    """
    Run inference on the test data using the trained model.

    Args:
        model (Model): Trained anomaly detection model.
        test_data (pd.DataFrame): Test data for inference.

    Returns:
        np.array: Predictions from the model.
    """
    logger.info(f"Running inference on {len(test_data)} samples.")
    
    try:
        predictions = model.predict(test_data)
        logger.info("Inference completed successfully.")
        return predictions
    except NotFittedError as e:
        logger.error(f"Model not fitted: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise e

def evaluate_predictions(test_data, predictions):
    """
    Evaluate the model's predictions.

    Args:
        test_data (pd.DataFrame): Ground truth test data.
        predictions (np.array): Model predictions.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    logger.info("Evaluating predictions...")
    
    try:
        eval_metrics = metrics.evaluate(test_data, predictions)
        logger.info(f"Evaluation results: {eval_metrics}")
        return eval_metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise e

def save_results(results, output_path):
    """
    Save the results to a YAML file.

    Args:
        results (dict): Evaluation metrics to save.
        output_path (str): Path where the results should be saved.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            yaml.dump(results, file)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise e

def run_experiment(config):
    """
    Run the experiment by loading data, initializing model, training, inference, and evaluation.

    Args:
        config (dict): Experiment configuration.
    """
    logger.info("Starting experiment...")
    
    validate_config(config)

    logger.info("Loading data...")
    data = data_loader.load_data(config['data']['path'])
    
    train_data, test_data = split_data(data, config['data']['test_size'])

    model = initialize_model(config)

    train_model(model, train_data)

    predictions = run_inference(model, test_data)

    eval_metrics = evaluate_predictions(test_data, predictions)

    return eval_metrics

def main(args):
    """
    Main entry point for the experiment script.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    logger.info(f"Loading configuration from {args.config}")
    
    config = load_config(args.config)
    
    results = run_experiment(config)

    results_path = os.path.join(config['output']['path'], 'results.yaml')
    
    save_results(results, results_path)
    
    logger.info("Experiment completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection experiment")
    
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()

    main(args)