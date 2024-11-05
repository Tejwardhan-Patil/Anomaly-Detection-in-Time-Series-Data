import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the time-series data
def load_data(filepath, sequence_length, split_fraction):
    """
    Load and preprocess time-series data for LSTM model.

    :param filepath: Path to the time-series data file (CSV format)
    :param sequence_length: Length of the sequences for LSTM
    :param split_fraction: Fraction of the data to use for training
    :return: Scaled training and test sets, along with the scaler
    """
    data = pd.read_csv(filepath)
    data_values = data.values

    # Scaling the data to (0, 1) range using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    sequence_data = []
    for i in range(len(scaled_data) - sequence_length):
        sequence_data.append(scaled_data[i: i + sequence_length])

    sequence_data = np.array(sequence_data)

    # Splitting data into training and testing sets
    train_size = int(len(sequence_data) * split_fraction)
    train_data = sequence_data[:train_size]
    test_data = sequence_data[train_size:]

    # Preparing X (input) and y (target) sets
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    return X_train, y_train, X_test, y_test, scaler

# Visualize the time-series data
def plot_time_series(data, title="Time-Series Data", xlabel="Time", ylabel="Value"):
    """
    Plot the time-series data.

    :param data: DataFrame or array-like object containing time-series data
    :param title: Title of the plot
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Build the LSTM model
def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.

    :param input_shape: Shape of the input data (sequence length, number of features)
    :return: Compiled LSTM model
    """
    model = Sequential()

    # First LSTM layer with 64 units and return sequences
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout for regularization

    # Second LSTM layer with 64 units
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))  # Dropout for regularization

    # Dense output layer with a single unit for regression
    model.add(Dense(units=1))

    # Compiling the model with Adam optimizer and MSE loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Train the LSTM model
def train_model(model, X_train, y_train, epochs=20, batch_size=64, validation_split=0.2):
    """
    Train the LSTM model on the training data.

    :param model: Compiled LSTM model
    :param X_train: Input data for training (sequences)
    :param y_train: Target data for training
    :param epochs: Number of epochs to train the model
    :param batch_size: Batch size for training
    :param validation_split: Fraction of the training data to use for validation
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Plot training and validation loss over epochs
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the LSTM model on the test data.

    :param model: Trained LSTM model
    :param X_test: Input data for testing
    :param y_test: Target data for testing
    :return: Test loss
    """
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    return test_loss

# Predict future values using the trained model
def predict_values(model, X_test, scaler):
    """
    Use the trained LSTM model to predict future values.

    :param model: Trained LSTM model
    :param X_test: Input data for testing (sequences)
    :param scaler: Scaler used to scale the input data
    :return: Inverse transformed predicted values
    """
    predictions = model.predict(X_test)
    predicted_values = scaler.inverse_transform(predictions)
    return predicted_values

# Plot the actual vs predicted values
def plot_predictions(y_test, predictions, title="Actual vs Predicted Values"):
    """
    Plot the actual vs predicted values.

    :param y_test: Actual target values (test set)
    :param predictions: Predicted values
    :param title: Title of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='Actual')
    plt.plot(predictions, color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Detect anomalies based on prediction errors
def detect_anomalies(model, X_test, y_test, scaler, threshold=0.01):
    """
    Detect anomalies by comparing actual and predicted values.

    :param model: Trained LSTM model
    :param X_test: Input data for testing (sequences)
    :param y_test: Target data for testing
    :param scaler: Scaler used to scale the input data
    :param threshold: Threshold for determining anomalies based on error
    :return: Array of anomaly flags (True for anomaly, False otherwise)
    """
    predictions = model.predict(X_test)
    predicted_values = scaler.inverse_transform(predictions)
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate the error between actual and predicted values
    error = np.abs(predicted_values - actual_values)

    # Flag anomalies where the error exceeds the threshold
    anomalies = error > threshold
    return anomalies

# Plot the detected anomalies
def plot_anomalies(actual_values, anomalies, title="Detected Anomalies"):
    """
    Plot the actual values and highlight detected anomalies.

    :param actual_values: Array of actual target values
    :param anomalies: Array of anomaly flags (True for anomaly, False otherwise)
    :param title: Title of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, color='blue', label='Actual Values')
    plt.scatter(np.where(anomalies), actual_values[anomalies], color='red', label='Anomalies')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Save the trained LSTM model
def save_model(model, model_path):
    """
    Save the trained LSTM model to a file.

    :param model: Trained LSTM model
    :param model_path: Path to save the model
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Load a saved LSTM model
def load_saved_model(model_path):
    """
    Load a previously saved LSTM model from a file.

    :param model_path: Path to the saved model
    :return: Loaded LSTM model
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

# Function to calculate anomaly score based on prediction error
def calculate_anomaly_score(y_true, y_pred):
    """
    Calculate anomaly scores based on the difference between actual and predicted values.

    :param y_true: Array of actual values
    :param y_pred: Array of predicted values
    :return: Anomaly scores (absolute error)
    """
    anomaly_scores = np.abs(y_true - y_pred)
    return anomaly_scores

# Identify anomaly windows
def identify_anomaly_windows(anomalies, window_size=5):
    """
    Identify windows of consecutive anomalies.

    :param anomalies: Array of anomaly flags (True for anomaly, False otherwise)
    :param window_size: Minimum size of consecutive anomalies to flag as an anomaly window
    :return: List of anomaly windows (start and end indices)
    """
    anomaly_windows = []
    start = None

    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly and start is None:
            start = i
        elif not is_anomaly and start is not None:
            if i - start >= window_size:
                anomaly_windows.append((start, i))
            start = None

    if start is not None and len(anomalies) - start >= window_size:
        anomaly_windows.append((start, len(anomalies)))

    return anomaly_windows

# Function to plot anomaly windows
def plot_anomaly_windows(actual_values, anomaly_windows, title="Anomaly Windows"):
    """
    Plot the time-series data and highlight anomaly windows.

    :param actual_values: Array of actual target values
    :param anomaly_windows: List of anomaly windows (start and end indices)
    :param title: Title of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, color='blue', label='Actual Values')

    for start, end in anomaly_windows:
        plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Window' if start == anomaly_windows[0][0] else "")

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Detect and visualize anomalies
def detect_and_visualize_anomalies(model, X_test, y_test, scaler, threshold=0.01, window_size=5):
    """
    Detect anomalies in the test data and visualize them, including anomaly windows.

    :param model: Trained LSTM model
    :param X_test: Input data for testing (sequences)
    :param y_test: Target data for testing
    :param scaler: Scaler used to scale the input data
    :param threshold: Threshold for determining anomalies based on error
    :param window_size: Minimum size of consecutive anomalies to flag as anomaly windows
    """
    predictions = model.predict(X_test)
    predicted_values = scaler.inverse_transform(predictions)
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate error and detect anomalies
    error = np.abs(predicted_values - actual_values)
    anomalies = error > threshold

    # Identify anomaly windows
    anomaly_windows = identify_anomaly_windows(anomalies, window_size)

    # Plot actual vs predicted values and anomalies
    plot_predictions(actual_values, predicted_values, title="Actual vs Predicted with Anomalies")
    plot_anomalies(actual_values, anomalies, title="Detected Anomalies")
    plot_anomaly_windows(actual_values, anomaly_windows, title="Detected Anomaly Windows")

# Main function to run the entire LSTM anomaly detection pipeline
def run_lstm_anomaly_detection_pipeline(data_filepath, model_save_path, sequence_length=60, split_fraction=0.8, epochs=20, batch_size=64, threshold=0.01, window_size=5):
    """
    Run the complete LSTM anomaly detection pipeline, including loading data, training, saving model, and detecting anomalies.

    :param data_filepath: Path to the time-series data file
    :param model_save_path: Path to save the trained model
    :param sequence_length: Length of the sequences for LSTM
    :param split_fraction: Fraction of the data to use for training
    :param epochs: Number of epochs to train the model
    :param batch_size: Batch size for training
    :param threshold: Threshold for detecting anomalies based on error
    :param window_size: Minimum size of consecutive anomalies to flag as anomaly windows
    """
    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_data(data_filepath, sequence_length, split_fraction)

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train, epochs, batch_size)

    # Save the trained model
    save_model(model, model_save_path)

    # Detect and visualize anomalies
    detect_and_visualize_anomalies(model, X_test, y_test, scaler, threshold, window_size)

# Function to load a model and run inference
def load_model_and_infer(model_path, data_filepath, sequence_length=60, split_fraction=0.8, threshold=0.01, window_size=5):
    """
    Load a saved model and run inference on new time-series data.

    :param model_path: Path to the saved model
    :param data_filepath: Path to the new time-series data file
    :param sequence_length: Length of the sequences for LSTM
    :param split_fraction: Fraction of the data to use for training
    :param threshold: Threshold for detecting anomalies based on error
    :param window_size: Minimum size of consecutive anomalies to flag as anomaly windows
    """
    # Load and preprocess data
    _, _, X_test, y_test, scaler = load_data(data_filepath, sequence_length, split_fraction)

    # Load the trained LSTM model
    model = load_saved_model(model_path)

    # Detect and visualize anomalies
    detect_and_visualize_anomalies(model, X_test, y_test, scaler, threshold, window_size)

# Function for hyperparameter tuning
def hyperparameter_tuning(data_filepath, sequence_length, split_fraction, epochs_list, batch_size_list, threshold, window_size):
    """
    Perform hyperparameter tuning for the LSTM model by testing different combinations of epochs and batch sizes.

    :param data_filepath: Path to the time-series data file
    :param sequence_length: Length of the sequences for LSTM
    :param split_fraction: Fraction of the data to use for training
    :param epochs_list: List of epoch values to try
    :param batch_size_list: List of batch size values to try
    :param threshold: Threshold for detecting anomalies based on error
    :param window_size: Minimum size of consecutive anomalies to flag as anomaly windows
    """
    best_loss = float('inf')
    best_params = {}

    for epochs in epochs_list:
        for batch_size in batch_size_list:
            print(f"Training with epochs={epochs} and batch_size={batch_size}...")
            X_train, y_train, X_test, y_test, scaler = load_data(data_filepath, sequence_length, split_fraction)

            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            train_model(model, X_train, y_train, epochs, batch_size)

            loss = evaluate_model(model, X_test, y_test)
            if loss < best_loss:
                best_loss = loss
                best_params = {'epochs': epochs, 'batch_size': batch_size}

    print(f"Best parameters: {best_params}, Best loss: {best_loss}")

# Function to perform cross-validation on the LSTM model
def cross_validate_lstm(data_filepath, sequence_length, split_fraction, epochs, batch_size, n_splits=5):
    """
    Perform cross-validation on the LSTM model using k-folds.

    :param data_filepath: Path to the time-series data file
    :param sequence_length: Length of the sequences for LSTM
    :param split_fraction: Fraction of the data to use for training
    :param epochs: Number of epochs to train the model
    :param batch_size: Batch size for training
    :param n_splits: Number of folds for cross-validation
    :return: Average test loss across folds
    """
    from sklearn.model_selection import KFold

    # Load and preprocess the data
    X_train, y_train, _, _, scaler = load_data(data_filepath, sequence_length, split_fraction)

    kfold = KFold(n_splits=n_splits, shuffle=True)
    fold_losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train)):
        print(f"Training fold {fold + 1}/{n_splits}...")

        # Split data into training and validation sets
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

        # Build and train the model
        model = build_lstm_model((X_fold_train.shape[1], X_fold_train.shape[2]))
        train_model(model, X_fold_train, y_fold_train, epochs, batch_size, validation_split=0)

        # Evaluate the model on the validation set
        val_loss = model.evaluate(X_fold_val, y_fold_val)
        fold_losses.append(val_loss)

    # Calculate average loss across folds
    avg_loss = np.mean(fold_losses)
    print(f"Average validation loss across {n_splits} folds: {avg_loss}")

    return avg_loss

# Function to add noise to the time-series data for robustness testing
def add_noise_to_data(data, noise_factor=0.05):
    """
    Add random noise to time-series data for robustness testing.

    :param data: Original time-series data
    :param noise_factor: Magnitude of noise to add
    :return: Data with added noise
    """
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    noisy_data = data + noise
    return noisy_data

# Function to test model robustness with noisy data
def test_model_robustness(model, X_test, y_test, scaler, noise_factor=0.05, threshold=0.01):
    """
    Test the robustness of the model by evaluating its performance on noisy data.

    :param model: Trained LSTM model
    :param X_test: Input data for testing (sequences)
    :param y_test: Target data for testing
    :param scaler: Scaler used to scale the input data
    :param noise_factor: Magnitude of noise to add to the test data
    :param threshold: Threshold for detecting anomalies based on error
    :return: Array of anomaly flags for the noisy data
    """
    # Add noise to the test data
    X_test_noisy = add_noise_to_data(X_test, noise_factor)

    # Predict using the model on noisy data
    predictions_noisy = model.predict(X_test_noisy)
    predicted_values_noisy = scaler.inverse_transform(predictions_noisy)

    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate error and detect anomalies
    error_noisy = np.abs(predicted_values_noisy - actual_values)
    anomalies_noisy = error_noisy > threshold

    return anomalies_noisy

# Plot performance on noisy data
def plot_robustness_test(actual_values, predictions_noisy, anomalies_noisy, title="Robustness Test: Actual vs Predicted with Noise"):
    """
    Plot the actual vs predicted values and detected anomalies for noisy data.

    :param actual_values: Array of actual target values
    :param predictions_noisy: Array of predicted values for noisy data
    :param anomalies_noisy: Array of anomaly flags for noisy data
    :param title: Title of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, color='blue', label='Actual Values')
    plt.plot(predictions_noisy, color='orange', label='Predicted Values (Noisy Data)')
    plt.scatter(np.where(anomalies_noisy), actual_values[anomalies_noisy], color='red', label='Anomalies')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to generate synthetic time-series data for testing purposes
def generate_synthetic_data(num_samples=1000, num_features=1, trend_factor=0.01, noise_factor=0.05):
    """
    Generate synthetic time-series data with a simple upward trend and added noise.

    :param num_samples: Number of time-series samples to generate
    :param num_features: Number of features in the time-series data
    :param trend_factor: Magnitude of the upward trend
    :param noise_factor: Magnitude of noise to add
    :return: Synthetic time-series data as a NumPy array
    """
    time = np.arange(num_samples)
    trend = trend_factor * time
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=(num_samples, num_features))

    synthetic_data = trend.reshape(-1, 1) + noise
    return synthetic_data

# Visualize synthetic time-series data
def plot_synthetic_data(synthetic_data, title="Synthetic Time-Series Data"):
    """
    Plot synthetic time-series data.

    :param synthetic_data: Array of synthetic time-series data
    :param title: Title of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(synthetic_data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Function to detect anomalies in synthetic data
def detect_anomalies_in_synthetic_data(model, synthetic_data, sequence_length, threshold=0.01):
    """
    Detect anomalies in synthetic time-series data using the LSTM model.

    :param model: Trained LSTM model
    :param synthetic_data: Synthetic time-series data
    :param sequence_length: Length of the sequences for LSTM
    :param threshold: Threshold for detecting anomalies based on error
    :return: Array of anomaly flags for the synthetic data
    """
    # Reshape synthetic data to fit LSTM input requirements
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_synthetic_data = scaler.fit_transform(synthetic_data)

    X_synthetic = []
    for i in range(len(scaled_synthetic_data) - sequence_length):
        X_synthetic.append(scaled_synthetic_data[i: i + sequence_length])

    X_synthetic = np.array(X_synthetic)

    # Predict using the model on synthetic data
    predictions_synthetic = model.predict(X_synthetic)
    predicted_values_synthetic = scaler.inverse_transform(predictions_synthetic)

    actual_values = synthetic_data[sequence_length:].reshape(-1, 1)

    # Calculate error and detect anomalies
    error_synthetic = np.abs(predicted_values_synthetic - actual_values)
    anomalies_synthetic = error_synthetic > threshold

    return anomalies_synthetic, actual_values

# Main function to generate synthetic data, detect anomalies, and visualize the results
def run_synthetic_anomaly_detection_pipeline(model, sequence_length=60, num_samples=1000, num_features=1, trend_factor=0.01, noise_factor=0.05, threshold=0.01):
    """
    Run the anomaly detection pipeline on synthetic time-series data.

    :param model: Trained LSTM model
    :param sequence_length: Length of the sequences for LSTM
    :param num_samples: Number of time-series samples to generate
    :param num_features: Number of features in the time-series data
    :param trend_factor: Magnitude of the upward trend
    :param noise_factor: Magnitude of noise to add
    :param threshold: Threshold for detecting anomalies based on error
    """
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(num_samples, num_features, trend_factor, noise_factor)

    # Detect anomalies in the synthetic data
    anomalies_synthetic, actual_values = detect_anomalies_in_synthetic_data(model, synthetic_data, sequence_length, threshold)

    # Plot synthetic data and detected anomalies
    plot_synthetic_data(synthetic_data, title="Generated Synthetic Data")
    plot_anomalies(actual_values, anomalies_synthetic, title="Anomalies in Synthetic Data")

# Usage
if __name__ == '__main__':
    # Paths and settings
    model_path = 'models/saved_lstm_model.h5'
    data_path = 'data/processed/time_series_data.csv'

    # Run LSTM anomaly detection pipeline on data
    run_lstm_anomaly_detection_pipeline(data_path, model_path, sequence_length=60, split_fraction=0.8, epochs=20, batch_size=64, threshold=0.01, window_size=5)

    # Load saved model and run inference on new data
    load_model_and_infer(model_path, data_path, sequence_length=60, split_fraction=0.8, threshold=0.01, window_size=5)

    # Generate synthetic data and detect anomalies
    lstm_model = load_saved_model(model_path)
    run_synthetic_anomaly_detection_pipeline(lstm_model, sequence_length=60, num_samples=1000, num_features=1, trend_factor=0.01, noise_factor=0.05, threshold=0.01)