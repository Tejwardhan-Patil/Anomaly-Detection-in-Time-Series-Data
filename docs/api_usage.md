# API Usage

## Overview

The anomaly detection model is accessible through a REST API. This section explains how to interact with the API to detect anomalies in time-series data.

### 1. Endpoints

- **POST /predict**: Submit time-series data and receive anomaly predictions.
  - Input: JSON object with time-series data.
  - Output: JSON object with anomaly scores and detected anomalies.

### 2. Python API

The Python API is built using Flask and can be found in `app.py`. It supports the following dependencies:

- `requirements.txt`: Lists the Python libraries required for the API.

### 3. R API

For R-based models, the API is built using Plumber, and the dependencies are listed in `packages.R`.

### Request

```bash
curl -X POST http://website.com/predict -H "Content-Type: application/json" -d '{
  "data": [1.2, 2.3, 3.4, 4.5]
}'
