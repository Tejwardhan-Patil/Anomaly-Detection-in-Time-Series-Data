import numpy as np
import pandas as pd
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Importing individual models
from models.traditional.arima import ARIMAModel
from models.traditional.random_forest import RandomForestModel
from models.deep_learning.lstm import LSTMModel
from models.deep_learning.autoencoder import AutoencoderModel
from models.unsupervised.isolation_forest import IsolationForestModel
from models.unsupervised.dbscan import DBSCANModel

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnsembleModel:
    def __init__(self, model_configs, voting='average', weights=None):
        """
        Initializes the ensemble model with different individual models and sets the aggregation strategy.
        model_configs: Dictionary with model initialization configurations.
        voting: Aggregation strategy: 'average' or 'majority'.
        weights: List of weights for weighted average voting (only used if voting='average').
        """
        self.models = {
            "arima": ARIMAModel(model_configs.get('arima', {})),
            "random_forest": RandomForestModel(model_configs.get('random_forest', {})),
            "lstm": LSTMModel(model_configs.get('lstm', {})),
            "autoencoder": AutoencoderModel(model_configs.get('autoencoder', {})),
            "isolation_forest": IsolationForestModel(model_configs.get('isolation_forest', {})),
            "dbscan": DBSCANModel(model_configs.get('dbscan', {}))
        }
        self.voting = voting
        self.weights = weights
        if voting == 'average' and weights is None:
            self.weights = [1.0] * len(self.models)  # Equal weighting by default

    def fit(self, X_train, y_train):
        """
        Fit each model on the training data.
        """
        for model_name, model in self.models.items():
            logging.info(f"Training {model_name} model...")
            try:
                if model_name != "dbscan":  # DBSCAN does not require supervised training
                    model.fit(X_train, y_train)
            except Exception as e:
                logging.error(f"Failed to train {model_name}: {str(e)}")

    def predict(self, X_test):
        """
        Make predictions using each model and aggregate them according to the selected strategy.
        For DBSCAN, outliers labeled as -1 are considered anomalies.
        """
        model_predictions = {}
        for model_name, model in self.models.items():
            logging.info(f"Generating predictions using {model_name} model...")
            try:
                if model_name == "dbscan":
                    pred = model.fit_predict(X_test)  # DBSCAN uses fit_predict instead of predict
                    pred = np.where(pred == -1, 1, 0)  # Treat -1 as anomalies (1), others as normal (0)
                else:
                    pred = model.predict(X_test)
                model_predictions[model_name] = pred
            except Exception as e:
                logging.error(f"Failed to predict using {model_name}: {str(e)}")
                model_predictions[model_name] = np.zeros(X_test.shape[0])

        predictions = np.array([model_predictions[model_name] for model_name in self.models])

        # Use the selected voting strategy to aggregate the predictions
        if self.voting == 'majority':
            logging.info("Applying majority voting for ensemble predictions.")
            ensemble_prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), axis=0, arr=predictions)
        elif self.voting == 'average':
            logging.info("Applying weighted average voting for ensemble predictions.")
            weighted_preds = np.average(predictions, axis=0, weights=self.weights)
            ensemble_prediction = (weighted_preds > 0.5).astype(int)
        else:
            raise ValueError("Voting strategy not recognized. Use 'majority' or 'average'.")
        
        return ensemble_prediction

    def evaluate(self, X_test, y_true):
        """
        Evaluate the ensemble model using various metrics.
        """
        logging.info("Evaluating ensemble model performance.")
        y_pred = self.predict(X_test)

        metrics = {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)
        }
        for metric, value in metrics.items():
            logging.info(f"{metric.capitalize()}: {value:.4f}")

        return metrics

    def save_predictions(self, X_test, output_path):
        """
        Save the ensemble predictions to a CSV file.
        """
        logging.info(f"Saving predictions to {output_path}.")
        y_pred = self.predict(X_test)
        prediction_df = pd.DataFrame(y_pred, columns=["Predictions"])
        prediction_df.to_csv(output_path, index=False)
        logging.info("Predictions saved successfully.")

    def save_model(self, model_name, save_path):
        """
        Method to save individual models.
        """
        logging.info(f"Saving {model_name} model to {save_path}.")
        model = self.models.get(model_name)
        if model is not None:
            try:
                model.save(save_path)
                logging.info(f"{model_name} model saved successfully.")
            except Exception as e:
                logging.error(f"Failed to save {model_name} model: {str(e)}")
        else:
            logging.error(f"Model {model_name} not found.")

if __name__ == "__main__":
    # Config dict for initializing models
    model_configs = {
        'arima': {'order': (5, 1, 0)},
        'random_forest': {'n_estimators': 100, 'max_depth': 10},
        'lstm': {'epochs': 50, 'batch_size': 32},
        'autoencoder': {'latent_dim': 16},
        'isolation_forest': {'n_estimators': 100},
        'dbscan': {'eps': 0.5, 'min_samples': 5}
    }

    # Load and split dataset (the dataset has already been preprocessed)
    logging.info("Loading dataset...")
    data = pd.read_csv('data/processed_data.csv')
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # Last column as the target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the ensemble model
    ensemble_model = EnsembleModel(model_configs, voting='average', weights=[1, 1, 2, 1, 1, 1])
    ensemble_model.fit(X_train, y_train)

    # Evaluate the model
    evaluation_metrics = ensemble_model.evaluate(X_test, y_test)
    logging.info(f"Final Evaluation Metrics: {evaluation_metrics}")

    # Save predictions to file
    ensemble_model.save_predictions(X_test, 'data/ensemble_predictions.csv')

    # Save individual models
    ensemble_model.save_model('lstm', 'saved_models/lstm_model.h5')
    ensemble_model.save_model('random_forest', 'saved_models/random_forest_model.pkl')