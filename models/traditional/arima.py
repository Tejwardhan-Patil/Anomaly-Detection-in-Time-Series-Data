import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import logging


class ARIMAAnomalyDetector:
    def __init__(self, order=(5, 1, 0), threshold=3.0, log_file='arima_anomaly.log'):
        """
        Initialize ARIMA-based anomaly detection model.

        :param order: Tuple of (p, d, q) for ARIMA model.
        :param threshold: Z-score threshold for anomaly detection.
        :param log_file: Path to log file.
        """
        self.order = order
        self.threshold = threshold
        self.model = None
        self.residuals = None
        self.model_fit = None
        self.data = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def fit(self, data: pd.Series):
        """
        Fit ARIMA model to time-series data.

        :param data: Time-series data as a Pandas Series.
        """
        try:
            self.logger.info('Fitting ARIMA model...')
            self.model = ARIMA(data, order=self.order)
            self.model_fit = self.model.fit()
            self.residuals = self.model_fit.resid
            self.data = data
            self.logger.info('ARIMA model fitted successfully.')
        except Exception as e:
            self.logger.error(f'Error in fitting ARIMA model: {e}')
            raise e

    def predict(self, start: int, end: int) -> pd.Series:
        """
        Generate predictions for given time range.

        :param start: Start index for prediction.
        :param end: End index for prediction.
        :return: Predicted values as a Pandas Series.
        """
        try:
            self.logger.info(f'Predicting values from {start} to {end}...')
            predictions = self.model_fit.predict(start=start, end=end)
            self.logger.info('Predictions generated successfully.')
            return predictions
        except Exception as e:
            self.logger.error(f'Error in prediction: {e}')
            raise e

    def detect_anomalies(self) -> pd.Series:
        """
        Detect anomalies based on residuals from ARIMA model.

        :return: Data points flagged as anomalies.
        """
        if self.residuals is None:
            self.logger.error('Model must be fitted before anomaly detection.')
            raise ValueError("Model must be fitted before anomaly detection.")
        
        try:
            self.logger.info('Detecting anomalies...')
            residuals_std = np.std(self.residuals)
            residuals_mean = np.mean(self.residuals)
            z_scores = (self.residuals - residuals_mean) / residuals_std
            anomalies = self.data[np.abs(z_scores) > self.threshold]
            self.logger.info(f'Anomalies detected: {len(anomalies)} found.')
            return anomalies
        except Exception as e:
            self.logger.error(f'Error in detecting anomalies: {e}')
            raise e

    def evaluate(self, actual: pd.Series, start: int, end: int) -> float:
        """
        Evaluate model performance using Mean Squared Error (MSE).

        :param actual: Actual time-series data for evaluation.
        :param start: Start index for evaluation.
        :param end: End index for evaluation.
        :return: Mean squared error between predictions and actual data.
        """
        try:
            predictions = self.predict(start=start, end=end)
            mse = mean_squared_error(actual[start:end], predictions)
            self.logger.info(f'Evaluation completed with MSE: {mse}')
            return mse
        except Exception as e:
            self.logger.error(f'Error during evaluation: {e}')
            raise e

    def save_model(self, file_path: str):
        """
        Save the fitted ARIMA model to disk.

        :param file_path: Path to save the model.
        """
        try:
            import joblib
            self.logger.info(f'Saving model to {file_path}...')
            joblib.dump(self.model_fit, file_path)
            self.logger.info('Model saved successfully.')
        except Exception as e:
            self.logger.error(f'Error saving model: {e}')
            raise e

    def load_model(self, file_path: str):
        """
        Load a previously saved ARIMA model.

        :param file_path: Path to load the model from.
        """
        try:
            import joblib
            self.logger.info(f'Loading model from {file_path}...')
            self.model_fit = joblib.load(file_path)
            self.residuals = self.model_fit.resid
            self.logger.info('Model loaded successfully.')
        except Exception as e:
            self.logger.error(f'Error loading model: {e}')
            raise e

    def plot_residuals(self):
        """
        Plot residuals of the ARIMA model to visualize fit.
        """
        try:
            import matplotlib.pyplot as plt
            self.logger.info('Plotting residuals...')
            plt.figure(figsize=(10, 6))
            plt.plot(self.residuals)
            plt.title('ARIMA Model Residuals')
            plt.xlabel('Time')
            plt.ylabel('Residuals')
            plt.show()
            self.logger.info('Residuals plotted successfully.')
        except Exception as e:
            self.logger.error(f'Error plotting residuals: {e}')
            raise e

    def plot_predictions(self, start: int, end: int):
        """
        Plot actual data and predictions for comparison.

        :param start: Start index for plot.
        :param end: End index for plot.
        """
        try:
            import matplotlib.pyplot as plt
            self.logger.info(f'Plotting predictions from {start} to {end}...')
            predictions = self.predict(start, end)
            plt.figure(figsize=(10, 6))
            plt.plot(self.data[start:end], label='Actual')
            plt.plot(predictions, label='Predicted', color='red')
            plt.title('Actual vs Predicted')
            plt.legend()
            plt.show()
            self.logger.info('Predictions plotted successfully.')
        except Exception as e:
            self.logger.error(f'Error plotting predictions: {e}')
            raise e

    def summary(self):
        """
        Print summary of the ARIMA model.
        """
        try:
            self.logger.info('Displaying model summary...')
            summary = self.model_fit.summary()
            print(summary)
            self.logger.info('Model summary displayed successfully.')
        except Exception as e:
            self.logger.error(f'Error displaying summary: {e}')
            raise e

    def grid_search(self, data: pd.Series, p_values: list, d_values: list, q_values: list):
        """
        Perform grid search over ARIMA hyperparameters (p, d, q).

        :param data: Time-series data as Pandas Series.
        :param p_values: List of p values for ARIMA.
        :param d_values: List of d values for ARIMA.
        :param q_values: List of q values for ARIMA.
        :return: Best combination of (p, d, q) values.
        """
        self.logger.info('Starting grid search for ARIMA hyperparameters...')
        best_score, best_cfg = float("inf"), None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        temp_model = ARIMA(data, order=(p, d, q))
                        temp_fit = temp_model.fit()
                        residuals = temp_fit.resid
                        mse = mean_squared_error(data, temp_fit.predict())
                        if mse < best_score:
                            best_score, best_cfg = mse, (p, d, q)
                        self.logger.info(f'Tested ARIMA({p},{d},{q}) with MSE={mse}')
                    except Exception as e:
                        self.logger.error(f'Error in ARIMA({p},{d},{q}): {e}')
                        continue
        self.logger.info(f'Best ARIMA configuration: {best_cfg} with MSE={best_score}')
        return best_cfg