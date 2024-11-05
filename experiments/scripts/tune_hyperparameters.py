import optuna
import yaml
import logging
from models.train import train_model
from models.evaluate import evaluate_model
from data.data_loader import load_data
from utils.metrics import custom_metrics
from time_series_utils import preprocess_data
from sklearn.model_selection import TimeSeriesSplit

# Initialize logger
logging.basicConfig(filename='hyperparameter_tuning.log', level=logging.INFO, 
                    format='%(asctime)s %(message)s')

# Load configuration file
def load_config():
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration file loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise

# Define objective function for Optuna
def objective(trial):
    try:
        # Load and preprocess data
        config = load_config()
        data = load_data(config['data']['train_path'])
        data = preprocess_data(data, config['preprocessing'])

        # Hyperparameter suggestions for different model types
        model_type = trial.suggest_categorical('model_type', ['lstm', 'gru', 'autoencoder'])
        
        # Shared hyperparameters
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        if model_type == 'lstm':
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            model_params = {
                'n_layers': n_layers,
                'hidden_units': hidden_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'bidirectional': bidirectional
            }
        elif model_type == 'gru':
            recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.1, 0.5)
            model_params = {
                'n_layers': n_layers,
                'hidden_units': hidden_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'recurrent_dropout': recurrent_dropout
            }
        else:  # autoencoder
            bottleneck_size = trial.suggest_int('bottleneck_size', 16, 64)
            model_params = {
                'n_layers': n_layers,
                'hidden_units': hidden_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'bottleneck_size': bottleneck_size
            }

        logging.info(f"Training with parameters: {model_params}")

        # Train model with suggested hyperparameters
        model = train_model(data, model_params, model_type=model_type)

        # Perform cross-validation
        tscv = TimeSeriesSplit(n_splits=config['cv']['n_splits'])
        cv_scores = []
        
        for train_index, val_index in tscv.split(data):
            train_data = data.iloc[train_index]
            val_data = data.iloc[val_index]
            
            val_data = preprocess_data(val_data, config['preprocessing'])
            predictions = model.predict(val_data)

            score = custom_metrics(val_data['target'], predictions)
            cv_scores.append(score)

        avg_score = sum(cv_scores) / len(cv_scores)
        logging.info(f"Cross-validation scores: {cv_scores}, Average Score: {avg_score}")
        
        return avg_score

    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}")
        raise

# Perform hyperparameter tuning using Optuna
def tune_hyperparameters():
    try:
        study_name = "anomaly_detection_study"  # Unique identifier of the study
        storage = "sqlite:///optuna_study.db"  # Save the study in a local SQLite database
        
        # Optuna study setup
        study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage, load_if_exists=True)
        
        # Optimize the objective function
        n_trials = load_config()['hyperparameters']['n_trials']
        logging.info(f"Starting hyperparameter optimization with {n_trials} trials.")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        # Save the best hyperparameters
        best_params = study.best_params
        with open('configs/best_params.yaml', 'w') as f:
            yaml.dump(best_params, f)
        logging.info(f"Best hyperparameters found: {best_params}")

        print("Best hyperparameters:", best_params)
        
    except Exception as e:
        logging.error(f"Error during hyperparameter optimization: {e}")
        raise

# Detailed function for loading and validating data
def validate_data(data):
    try:
        # Check for missing values
        if data.isnull().values.any():
            logging.warning("Data contains missing values. Applying imputation.")
            data.fillna(method='ffill', inplace=True)
        
        # Ensure there are enough samples for each time window
        if len(data) < 100:
            raise ValueError("Insufficient data points for time-series modeling.")
        
        logging.info("Data validation passed.")
        return data
    except Exception as e:
        logging.error(f"Data validation failed: {e}")
        raise

# Run hyperparameter tuning process
if __name__ == "__main__":
    try:
        logging.info("Hyperparameter tuning process started.")
        
        # Initial data loading
        config = load_config()
        data = load_data(config['data']['train_path'])
        data = preprocess_data(data, config['preprocessing'])
        data = validate_data(data)

        tune_hyperparameters()

        logging.info("Hyperparameter tuning process completed.")
    
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise