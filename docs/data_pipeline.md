# Data Pipeline

## Overview

The data pipeline processes raw time-series data and prepares it for anomaly detection models. This includes preprocessing, feature engineering, and data splitting.

### 1. Data Preprocessing

- **Preprocessing Scripts**: Raw data is cleaned and normalized using Python (`preprocess.py`) and R (`preprocess.R`). Missing values are handled, and data is resampled as necessary.

### 2. Feature Engineering

- **Feature Engineering Scripts**: Features such as moving averages, lag features, and Fourier transforms are generated using Python (`feature_engineering.py`). The features are stored in the `features/` directory for use in training models.

### 3. Data Splitting

- **Data Splitter**: Time-series data is split into training, validation, and test sets using a time-based split method. Implemented in Python (`split.py`).

The pipeline supports both Python and R environments and allows for seamless integration with the models.
