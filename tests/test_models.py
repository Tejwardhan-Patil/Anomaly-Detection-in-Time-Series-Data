import unittest
import numpy as np
import pandas as pd
from models.traditional.arima import ARIMA
from models.deep_learning.lstm import LSTMModel
from models.unsupervised.isolation_forest import IsolationForestModel
from models.ensemble.ensemble_model import EnsembleModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TestModels(unittest.TestCase):

    def setUp(self):
        # Create a sample time-series dataset with some missing values for testing
        self.data = pd.Series([1.1, 2.5, np.nan, 4.2, 5.8, np.nan, 7.3, 8.9])
        self.data_filled = self.data.fillna(method='ffill')  # Forward fill missing values
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Standardize and normalize the data
        self.standardized_data = self.scaler.fit_transform(self.data_filled.values.reshape(-1, 1))
        self.normalized_data = self.minmax_scaler.fit_transform(self.data_filled.values.reshape(-1, 1))

    def test_arima_model(self):
        # Initialize ARIMA model with different parameters
        arima_model = ARIMA(order=(1, 1, 1))
        arima_model.fit(self.data_filled)

        # Predict values and ensure there are no NaNs
        predictions = arima_model.predict(len(self.data))
        self.assertFalse(np.isnan(predictions).any(), "ARIMA model produced NaNs in predictions")
        self.assertEqual(len(predictions), len(self.data), "ARIMA prediction length mismatch")

    def test_lstm_model(self):
        # Initialize LSTM model
        lstm_model = LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)
        lstm_model.fit(self.data_filled)

        # Predict values and ensure there are no NaNs
        predictions = lstm_model.predict(self.data_filled)
        self.assertFalse(np.isnan(predictions).any(), "LSTM model produced NaNs in predictions")
        self.assertEqual(len(predictions), len(self.data_filled), "LSTM prediction length mismatch")

    def test_lstm_model_with_standardized_data(self):
        # Train LSTM model on standardized data
        lstm_model = LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)
        lstm_model.fit(self.standardized_data)

        predictions = lstm_model.predict(self.standardized_data)
        self.assertFalse(np.isnan(predictions).any(), "LSTM model produced NaNs with standardized data")

    def test_lstm_model_with_normalized_data(self):
        # Train LSTM model on normalized data
        lstm_model = LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)
        lstm_model.fit(self.normalized_data)

        predictions = lstm_model.predict(self.normalized_data)
        self.assertFalse(np.isnan(predictions).any(), "LSTM model produced NaNs with normalized data")

    def test_isolation_forest(self):
        # Initialize Isolation Forest model with different contamination levels
        iso_forest_model = IsolationForestModel(contamination=0.1)
        iso_forest_model.fit(self.data_filled.values.reshape(-1, 1))

        # Ensure the model detects outliers and no NaNs are produced
        anomalies = iso_forest_model.predict(self.data_filled.values.reshape(-1, 1))
        self.assertIn(-1, anomalies, "Isolation Forest failed to detect anomalies")

    def test_isolation_forest_standardized(self):
        # Test Isolation Forest on standardized data
        iso_forest_model = IsolationForestModel(contamination=0.1)
        iso_forest_model.fit(self.standardized_data)

        anomalies = iso_forest_model.predict(self.standardized_data)
        self.assertIn(-1, anomalies, "Isolation Forest failed to detect anomalies in standardized data")

    def test_isolation_forest_normalized(self):
        # Test Isolation Forest on normalized data
        iso_forest_model = IsolationForestModel(contamination=0.1)
        iso_forest_model.fit(self.normalized_data)

        anomalies = iso_forest_model.predict(self.normalized_data)
        self.assertIn(-1, anomalies, "Isolation Forest failed to detect anomalies in normalized data")

    def test_ensemble_model(self):
        # Initialize Ensemble model with both ARIMA and LSTM models
        ensemble_model = EnsembleModel(models=[ARIMA(order=(1, 1, 1)), LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)])
        ensemble_model.fit(self.data_filled)

        # Ensure the ensemble model runs and produces predictions without NaNs
        predictions = ensemble_model.predict(self.data_filled)
        self.assertFalse(np.isnan(predictions).any(), "Ensemble model produced NaNs in predictions")

    def test_ensemble_model_standardized(self):
        # Test Ensemble model on standardized data
        ensemble_model = EnsembleModel(models=[ARIMA(order=(1, 1, 1)), LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)])
        ensemble_model.fit(self.standardized_data)

        predictions = ensemble_model.predict(self.standardized_data)
        self.assertFalse(np.isnan(predictions).any(), "Ensemble model produced NaNs with standardized data")

    def test_ensemble_model_normalized(self):
        # Test Ensemble model on normalized data
        ensemble_model = EnsembleModel(models=[ARIMA(order=(1, 1, 1)), LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)])
        ensemble_model.fit(self.normalized_data)

        predictions = ensemble_model.predict(self.normalized_data)
        self.assertFalse(np.isnan(predictions).any(), "Ensemble model produced NaNs with normalized data")

    def test_data_shape_mismatch(self):
        # Test that models handle shape mismatch appropriately
        lstm_model = LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)
        
        # Intentionally using mismatched input shape
        with self.assertRaises(ValueError):
            lstm_model.fit(self.data.values.reshape(1, -1))

    def test_isolation_forest_with_anomalous_data(self):
        # Test Isolation Forest with a synthetic dataset containing anomalies
        anomalous_data = pd.Series([1.1, 2.5, 100.0, 4.2, 5.8, 200.0, 7.3, 8.9])
        iso_forest_model = IsolationForestModel(contamination=0.2)
        iso_forest_model.fit(anomalous_data.values.reshape(-1, 1))

        anomalies = iso_forest_model.predict(anomalous_data.values.reshape(-1, 1))
        self.assertEqual(list(anomalies).count(-1), 2, "Isolation Forest failed to detect correct number of anomalies")

    def test_ensemble_with_different_model_types(self):
        # Test ensemble with different types of models and data
        ensemble_model = EnsembleModel(models=[
            ARIMA(order=(1, 1, 1)),
            IsolationForestModel(contamination=0.1)
        ])
        ensemble_model.fit(self.data_filled)

        predictions = ensemble_model.predict(self.data_filled)
        self.assertFalse(np.isnan(predictions).any(), "Ensemble model with mixed models produced NaNs")

    def test_missing_data_handling(self):
        # Test missing data handling for ARIMA model
        arima_model = ARIMA(order=(1, 1, 1))
        with self.assertRaises(ValueError):
            arima_model.fit(self.data)  # Dataset with missing values

if __name__ == '__main__':
    unittest.main()