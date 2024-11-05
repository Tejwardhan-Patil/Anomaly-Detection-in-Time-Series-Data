import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils.data_loader import load_data
from utils.metrics import evaluate_model
import joblib
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(filename='logs/isolation_forest.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class IsolationForestModel:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )
        logging.info(f"Initialized IsolationForest with {n_estimators} trees, contamination set to {contamination}")

    def preprocess_data(self, X):
        """
        Standardize the features in X.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Data has been standardized.")
        return X_scaled

    def feature_engineering(self, X):
        """
        Apply feature engineering to the dataset (e.g., rolling statistics).
        """
        X['rolling_mean'] = X.rolling(window=5).mean()
        X['rolling_std'] = X.rolling(window=5).std()
        X.fillna(method='bfill', inplace=True)
        logging.info("Feature engineering applied: rolling mean and rolling std.")
        return X

    def train(self, X_train):
        X_train_preprocessed = self.preprocess_data(X_train)
        self.model.fit(X_train_preprocessed)
        logging.info("Isolation Forest model has been trained.")

    def predict(self, X_test):
        X_test_preprocessed = self.preprocess_data(X_test)
        predictions = self.model.predict(X_test_preprocessed)
        logging.info("Predictions made on the test set.")
        return predictions
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model using precision, recall, F1 score, and accuracy.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=-1)
        recall = recall_score(y_true, y_pred, pos_label=-1)
        f1 = f1_score(y_true, y_pred, pos_label=-1)

        logging.info(f"Evaluation metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    def plot_anomalies(self, X_test, y_pred):
        """
        Visualize anomalies in the test data.
        """
        anomaly_indices = np.where(y_pred == -1)[0]
        plt.figure(figsize=(12,6))
        plt.plot(X_test.index, X_test, label="Test Data")
        plt.scatter(X_test.index[anomaly_indices], X_test.iloc[anomaly_indices], 
                    color='red', label="Anomalies", marker='x')
        plt.title('Anomalies Detected by Isolation Forest')
        plt.legend()
        plt.show()
        logging.info("Anomalies plotted.")

    def save_model(self, path):
        joblib.dump(self.model, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path):
        self.model = joblib.load(path)
        logging.info(f"Model loaded from {path}")

def split_data(X, y, train_size=0.8):
    """
    Splits the data into training and testing sets.
    """
    split_point = int(train_size * len(X))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    logging.info(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test

def main():
    logging.info("Started Isolation Forest anomaly detection process.")
    
    # Load preprocessed time-series data
    data = load_data('data/processed/time_series_data.csv')
    
    # Data is structured with features and a target column
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Feature engineering
    model = IsolationForestModel()
    X = model.feature_engineering(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train the model
    model.train(X_train)
    
    # Predict on the test data
    predictions = model.predict(X_test)
    
    # Evaluate the model
    model.evaluate(y_test, predictions)
    
    # Visualize anomalies
    model.plot_anomalies(X_test, predictions)
    
    # Save the trained model
    model.save_model('models/saved/isolation_forest_model.pkl')
    
    logging.info("Isolation Forest anomaly detection process completed.")

if __name__ == "__main__":
    main()