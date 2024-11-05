import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(file_path):
    """
    Loads the time-series data from a CSV file, normalizes it using MinMaxScaler,
    and returns the scaled data along with the scaler object for inverse transformation.
    """
    try:
        data = np.loadtxt(file_path, delimiter=',')
        print(f"Data loaded from {file_path} with shape {data.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Split the dataset into training and test sets
def split_data(data, test_size=0.2):
    """
    Splits the data into training and test sets.
    """
    try:
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
        print(f"Data split into train and test sets. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        return None, None

# Build the autoencoder model
def build_autoencoder(input_dim, encoding_dim):
    """
    Builds an autoencoder model with the specified input dimension and encoding dimension.
    The model compresses the input data into a latent space and reconstructs it.
    """
    try:
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        print("Autoencoder model built successfully.")
        return autoencoder
    except Exception as e:
        print(f"Error building autoencoder: {str(e)}")
        return None

# Train the autoencoder model
def train_autoencoder(autoencoder, train_data, epochs=50, batch_size=32):
    """
    Trains the autoencoder on the given training data.
    """
    try:
        history = autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)
        print(f"Autoencoder training completed for {epochs} epochs.")
        return history
    except Exception as e:
        print(f"Error training autoencoder: {str(e)}")
        return None

# Plot the training loss and validation loss
def plot_training_history(history):
    """
    Plots the training and validation loss over epochs.
    """
    try:
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")

# Detect anomalies using reconstruction error
def detect_anomalies(autoencoder, data, threshold=0.01):
    """
    Detects anomalies in the data using reconstruction error.
    Returns a binary mask indicating whether each data point is an anomaly.
    """
    try:
        predictions = autoencoder.predict(data)
        mse = np.mean(np.power(data - predictions, 2), axis=1)
        anomalies = mse > threshold
        print(f"Anomalies detected: {np.sum(anomalies)} out of {len(anomalies)} data points.")
        return anomalies, mse
    except Exception as e:
        print(f"Error detecting anomalies: {str(e)}")
        return None, None

# Plot anomalies on a graph
def plot_anomalies(data, anomalies, mse):
    """
    Plots the time-series data and highlights the anomalies with a different color.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Time-Series Data')
        plt.scatter(np.where(anomalies), data[anomalies], color='r', label='Anomalies')
        plt.title('Detected Anomalies in Time-Series Data')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error plotting anomalies: {str(e)}")

# Function to save the autoencoder model
def save_model(autoencoder, model_path):
    """
    Saves the trained autoencoder model to the specified path.
    """
    try:
        autoencoder.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

# Function to load the autoencoder model
def load_model(model_path):
    """
    Loads a pre-trained autoencoder model from the specified path.
    """
    try:
        autoencoder = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return autoencoder
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    # Define the path to the time-series data and model save path
    file_path = 'data/processed/time_series_data.csv'
    model_path = 'models/autoencoder_model.h5'

    # Load and preprocess the data
    data, scaler = load_data(file_path)
    if data is None:
        raise ValueError("Failed to load data. Exiting.")

    # Split the data into training and test sets
    train_data, test_data = split_data(data)
    if train_data is None or test_data is None:
        raise ValueError("Failed to split data. Exiting.")

    # Define autoencoder parameters
    input_dim = train_data.shape[1]
    encoding_dim = input_dim // 2  # Compression to half of the input dimension

    # Build the autoencoder model
    autoencoder = build_autoencoder(input_dim, encoding_dim)
    if autoencoder is None:
        raise ValueError("Failed to build autoencoder model. Exiting.")

    # Train the autoencoder
    epochs = 100
    batch_size = 64

    history = train_autoencoder(autoencoder, train_data, epochs=epochs, batch_size=batch_size)
    if history is None:
        raise ValueError("Failed to train autoencoder model. Exiting.")

    # Plot the training and validation loss
    plot_training_history(history)

    # Save the trained autoencoder model
    save_model(autoencoder, model_path)

    # Load the autoencoder model for anomaly detection
    autoencoder_loaded = load_model(model_path)
    if autoencoder_loaded is None:
        raise ValueError("Failed to load autoencoder model. Exiting.")

    # Detect anomalies in the test data
    threshold = 0.02 
    anomalies, reconstruction_errors = detect_anomalies(autoencoder_loaded, test_data)
    if anomalies is None:
        raise ValueError("Failed to detect anomalies. Exiting.")

    # Plot the anomalies on the test data
    plot_anomalies(test_data[:, 0], anomalies, reconstruction_errors)

    # Calculate anomaly scores and display results
    def calculate_anomaly_scores(reconstruction_errors, threshold):
        """
        Calculates anomaly scores based on the reconstruction error.
        Any score above the threshold is considered an anomaly.
        """
        anomaly_scores = reconstruction_errors - threshold
        anomaly_scores = np.where(anomaly_scores > 0, anomaly_scores, 0)
        return anomaly_scores

    anomaly_scores = calculate_anomaly_scores(reconstruction_errors, threshold)
    print(f"Anomaly scores calculated for the test set. Max anomaly score: {np.max(anomaly_scores)}")

    # Evaluate model performance based on precision, recall, and F1-score
    from sklearn.metrics import precision_score, recall_score, f1_score

    def evaluate_model(anomalies, true_labels):
        """
        Evaluates the anomaly detection model by comparing predicted anomalies with true labels.
        Returns precision, recall, and F1-score metrics.
        """
        precision = precision_score(true_labels, anomalies)
        recall = recall_score(true_labels, anomalies)
        f1 = f1_score(true_labels, anomalies)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        return precision, recall, f1

    # 'true_labels' is a binary array indicating actual anomalies in the test set
    true_labels = np.zeros(test_data.shape[0])
    # Define the true anomaly points manually or via domain-specific rules
    true_labels[50:60] = 1  # Label data points 50 to 60 as anomalies

    precision, recall, f1 = evaluate_model(anomalies, true_labels)

    # Function to visualize reconstruction error
    def plot_reconstruction_error(reconstruction_errors):
        """
        Plots the reconstruction errors for each data point in the test set.
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(reconstruction_errors, label='Reconstruction Error')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            plt.title('Reconstruction Errors on Test Data')
            plt.xlabel('Data Point Index')
            plt.ylabel('Reconstruction Error')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error plotting reconstruction errors: {str(e)}")

    # Plot reconstruction errors for the test data
    plot_reconstruction_error(reconstruction_errors)

    # Function to inverse scale the anomalies for interpretation
    def inverse_transform_anomalies(scaler, anomalies, original_data):
        """
        Inversely transforms the detected anomalies to their original scale using the scaler object.
        """
        try:
            original_anomalies = scaler.inverse_transform(anomalies.reshape(-1, 1))
            print(f"Anomalies inversely transformed to original scale.")
            return original_anomalies
        except Exception as e:
            print(f"Error in inverse transformation: {str(e)}")
            return None

    # Inversely transform the anomalies for better interpretability
    inverse_anomalies = inverse_transform_anomalies(scaler, anomalies, test_data)
    if inverse_anomalies is not None:
        print(f"First few inversely transformed anomalies: {inverse_anomalies[:5].flatten()}")

    # Function to evaluate performance based on threshold tuning
    def tune_threshold(reconstruction_errors, true_labels):
        """
        Finds the optimal threshold by evaluating different thresholds and
        calculating precision, recall, and F1-score at each threshold.
        """
        thresholds = np.linspace(0.01, 0.1, num=100)
        best_threshold = threshold
        best_f1 = 0

        for t in thresholds:
            predicted_anomalies = reconstruction_errors > t
            precision, recall, f1 = evaluate_model(predicted_anomalies, true_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"Best Threshold: {best_threshold:.4f}, Best F1-Score: {best_f1:.4f}")
        return best_threshold

    # Tune the threshold based on reconstruction errors and true labels
    optimal_threshold = tune_threshold(reconstruction_errors, true_labels)

    # Save final model and results
    def save_final_model_and_results(autoencoder, model_path, anomaly_scores, file_path):
        """
        Saves the final trained autoencoder model and anomaly scores to disk.
        """
        try:
            autoencoder.save(model_path)
            print(f"Final model saved to {model_path}.")
            
            np.savetxt(file_path, anomaly_scores, delimiter=',')
            print(f"Anomaly scores saved to {file_path}.")
        except Exception as e:
            print(f"Error saving final model and results: {str(e)}")

    # Save the final model and anomaly scores
    save_final_model_and_results(autoencoder, 'models/final_autoencoder_model.h5', anomaly_scores, 'data/anomaly_scores.csv')

    # Define a function for loading the final saved model
    def load_final_model(model_path):
        """
        Loads the final saved autoencoder model from disk.
        """
        try:
            autoencoder = tf.keras.models.load_model(model_path)
            print(f"Final model loaded from {model_path}.")
            return autoencoder
        except Exception as e:
            print(f"Error loading final model: {str(e)}")
            return None

    # Load the final autoencoder model
    final_autoencoder = load_final_model('models/final_autoencoder_model.h5')
    if final_autoencoder is None:
        raise ValueError("Failed to load the final autoencoder model.")
    
    # Final predictions using the loaded model
    final_anomalies, final_reconstruction_errors = detect_anomalies(final_autoencoder, test_data)
    if final_anomalies is None:
        raise ValueError("Failed to perform final anomaly detection.")

    # Plot the final reconstruction errors
    plot_reconstruction_error(final_reconstruction_errors)

    # Function to generate a comprehensive anomaly detection report
    def generate_anomaly_report(anomalies, anomaly_scores, output_path):
        """
        Generates a detailed report of the detected anomalies, including their respective anomaly scores
        and timestamps or indices. Saves the report as a CSV file.
        """
        try:
            with open(output_path, 'w') as file:
                file.write("Index,Anomaly Score,Anomaly Detected\n")
                for i, (score, anomaly) in enumerate(zip(anomaly_scores, anomalies)):
                    file.write(f"{i},{score},{int(anomaly)}\n")
            print(f"Anomaly report generated and saved to {output_path}")
        except Exception as e:
            print(f"Error generating anomaly report: {str(e)}")

    # Generate the anomaly report
    anomaly_report_path = 'reports/anomaly_report.csv'
    generate_anomaly_report(final_anomalies, final_reconstruction_errors, anomaly_report_path)

    # Function to visualize true vs predicted anomalies
    def plot_true_vs_predicted(true_labels, predicted_anomalies):
        """
        Plots the true anomalies and predicted anomalies for visual comparison.
        """
        try:
            plt.figure(figsize=(12, 8))
            plt.plot(true_labels, label='True Anomalies', marker='o')
            plt.plot(predicted_anomalies, label='Predicted Anomalies', marker='x')
            plt.title('True vs Predicted Anomalies')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error plotting true vs predicted anomalies: {str(e)}")

    # Visualize the true vs predicted anomalies
    plot_true_vs_predicted(true_labels, final_anomalies)

    # Function to deploy the model as a REST API
    from flask import Flask, request, jsonify

    def create_api(autoencoder, scaler, threshold):
        """
        Creates a simple Flask API for serving the anomaly detection model.
        The API receives time-series data, processes it, and returns detected anomalies.
        """
        app = Flask(__name__)

        @app.route('/predict', methods=['POST'])
        def predict():
            """
            API endpoint for predicting anomalies.
            The input should be a JSON array of time-series data.
            """
            try:
                data = request.json['data']
                data = np.array(data)
                data_scaled = scaler.transform(data)

                # Predict anomalies
                predictions = autoencoder.predict(data_scaled)
                reconstruction_errors = np.mean(np.power(data_scaled - predictions, 2), axis=1)
                anomalies = reconstruction_errors > threshold

                response = {
                    'anomalies': anomalies.tolist(),
                    'reconstruction_errors': reconstruction_errors.tolist()
                }

                return jsonify(response)

            except Exception as e:
                return jsonify({'error': str(e)})

        return app

    # Create and run the API
    autoencoder_api = create_api(final_autoencoder, scaler, optimal_threshold)
    autoencoder_api.run(host='0.0.0.0', port=5000)

    # Function to test the deployed API with sample data
    import requests
    import json

    def test_api(api_url, sample_data):
        """
        Tests the deployed API by sending sample time-series data and retrieving the predicted anomalies.
        """
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(api_url, data=json.dumps({'data': sample_data.tolist()}), headers=headers)

            if response.status_code == 200:
                result = response.json()
                print(f"Anomalies: {result['anomalies']}")
                print(f"Reconstruction Errors: {result['reconstruction_errors'][:5]}")  # Display first 5 errors
            else:
                print(f"Error in API response: {response.text}")

        except Exception as e:
            print(f"Error testing API: {str(e)}")

    # Test the API with a sample from the test data
    api_url = 'http://127.0.0.1:5000/predict'
    test_api(api_url, test_data[:10])

    # Final function to evaluate performance on new incoming data
    def evaluate_on_new_data(new_data, autoencoder, scaler, threshold):
        """
        Evaluates the autoencoder's performance on new incoming data by detecting anomalies.
        Returns the detected anomalies and reconstruction errors.
        """
        try:
            new_data_scaled = scaler.transform(new_data)
            predictions = autoencoder.predict(new_data_scaled)
            reconstruction_errors = np.mean(np.power(new_data_scaled - predictions, 2), axis=1)
            anomalies = reconstruction_errors > threshold

            print(f"Detected {np.sum(anomalies)} anomalies in the new data.")
            return anomalies, reconstruction_errors

        except Exception as e:
            print(f"Error evaluating new data: {str(e)}")
            return None, None

    # Simulate new data for evaluation
    new_data = test_data[:50]
    evaluate_on_new_data(new_data, final_autoencoder, scaler, optimal_threshold)

    # Create a function to log detected anomalies
    import logging

    def setup_logging(log_file='anomaly_detection.log'):
        """
        Sets up logging for anomaly detection results.
        Logs detected anomalies, reconstruction errors, and relevant information.
        """
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log_anomalies(anomalies, reconstruction_errors):
        """
        Logs the detected anomalies and their corresponding reconstruction errors.
        """
        try:
            logging.info("Anomalies Detected:")
            for i, (anomaly, error) in enumerate(zip(anomalies, reconstruction_errors)):
                if anomaly:
                    logging.info(f"Index {i}: Reconstruction Error = {error}")

        except Exception as e:
            logging.error(f"Error logging anomalies: {str(e)}")

    # Set up logging and log the final anomalies
    setup_logging()
    log_anomalies(final_anomalies, final_reconstruction_errors)

    # Function to handle real-time anomaly detection (streaming data)
    def real_time_anomaly_detection(stream, autoencoder, scaler, threshold):
        """
        Handles real-time anomaly detection by processing a stream of incoming data.
        Continuously detects anomalies on incoming data batches.
        """
        try:
            for batch in stream:
                batch_scaled = scaler.transform(batch)
                predictions = autoencoder.predict(batch_scaled)
                reconstruction_errors = np.mean(np.power(batch_scaled - predictions, 2), axis=1)
                anomalies = reconstruction_errors > threshold

                print(f"Processed batch with {np.sum(anomalies)} anomalies detected.")
                log_anomalies(anomalies, reconstruction_errors)

        except Exception as e:
            logging.error(f"Error in real-time anomaly detection: {str(e)}")

    # Usage of real-time anomaly detection with simulated data stream
    simulated_stream = [test_data[i:i+10] for i in range(0, len(test_data), 10)]
    real_time_anomaly_detection(simulated_stream, final_autoencoder, scaler, optimal_threshold)