# Model Architectures

## Overview

This repository includes various model architectures for anomaly detection in time-series data. The models are categorized into traditional, deep learning, unsupervised, and ensemble approaches.

### 1. Traditional Models

- **ARIMA**: Autoregressive Integrated Moving Average (ARIMA) is a classical statistical model for time-series analysis. It is implemented in Python (`arima.py`) and also available in R using the `forecast` package (`r_models.R`).

- **SARIMA**: Seasonal ARIMA, an extension of ARIMA that supports seasonality. This model is implemented in R using the `forecast` package.

- **Random Forest**: A machine learning model that works well on time-series data by converting it into tabular form with features such as lag and rolling statistics. The implementation is available in Python (`random_forest.py`).

### 2. Deep Learning Models

- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) that is effective in detecting anomalies in sequences. The model is implemented in Python (`lstm.py`).

- **Autoencoder**: An unsupervised learning model designed to compress data and reconstruct it. When reconstruction error is high, the data point is flagged as an anomaly. Implemented in Python (`autoencoder.py`).

### 3. Unsupervised Models

- **Isolation Forest**: An algorithm specifically designed for anomaly detection by isolating outliers. Available in Python (`isolation_forest.py`).

- **DBSCAN**: A clustering algorithm that detects outliers based on density. Implemented in Python (`dbscan.py`).

### 4. Ensemble Models

Ensemble models combine predictions from multiple models to improve accuracy. These are implemented in Python (`ensemble_model.py`), and they support both traditional and deep learning models.
