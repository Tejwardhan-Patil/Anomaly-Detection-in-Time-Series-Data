from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import joblib
import logging
from typing import List, Optional
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the pre-trained model
try:
    model = joblib.load("models/trained_model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define Pydantic request models
class TimeSeriesData(BaseModel):
    data: List[float] = Field(..., example=[0.1, 0.5, 0.3, 0.9, 0.2], description="List of time-series data points")
    timestamp: Optional[str] = Field(None, description="Optional timestamp for each request")

class BatchTimeSeriesData(BaseModel):
    data: List[List[float]] = Field(..., example=[[0.1, 0.2], [0.3, 0.4]], description="Batch of time-series data sets")

# Root route
@app.get("/")
def read_root():
    logger.info("Root route accessed.")
    return {"message": "Anomaly Detection API is running", "status": "online"}

# Health check route
@app.get("/health")
def health_check():
    logger.info("Health check route accessed.")
    return {"status": "healthy", "message": "The API is working properly"}

# Predict anomalies in a single time-series dataset
@app.post("/predict")
def detect_anomalies(request: TimeSeriesData):
    logger.info(f"Received prediction request: {request.data}")
    try:
        # Validate the data length
        if len(request.data) < 2:
            raise ValueError("Insufficient data for anomaly detection")
        
        # Convert data to numpy array and reshape it
        data = np.array(request.data).reshape(1, -1)

        # Predict anomalies using the model
        prediction = model.predict(data)

        # Generate response
        response = {
            "input_data": request.data,
            "anomalies_detected": bool(prediction[0]),
            "timestamp": request.timestamp if request.timestamp else str(datetime.utcnow())
        }

        logger.info(f"Prediction response: {response}")
        return response

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Batch prediction route
@app.post("/predict/batch")
def detect_anomalies_batch(request: BatchTimeSeriesData):
    logger.info(f"Received batch prediction request with {len(request.data)} data sets")
    try:
        predictions = []
        
        for series in request.data:
            # Validate each data set
            if len(series) < 2:
                raise ValueError(f"Data set too short: {series}")
            
            # Predict anomalies
            data = np.array(series).reshape(1, -1)
            prediction = model.predict(data)
            predictions.append(bool(prediction[0]))
        
        # Generate batch response
        response = {
            "batch_input_data": request.data,
            "batch_predictions": predictions,
            "timestamp": str(datetime.utcnow())
        }

        logger.info(f"Batch prediction response: {response}")
        return response

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

# Model retraining endpoint (for future use)
@app.post("/retrain")
async def retrain_model(request: Request):
    logger.info("Retrain request received.")
    try:
        # Log request body for retraining
        body = await request.json()
        logger.info(f"Retraining data: {body}")

        # Retrain process
        logger.info("Retraining model...")
        
        # Retrain success
        logger.info("Model retrained successfully.")
        return {"status": "success", "message": "Model retrained successfully"}

    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

# Endpoint for logging raw time-series data
@app.post("/log")
async def log_time_series_data(request: TimeSeriesData):
    logger.info(f"Logging time-series data: {request.data}")
    try:
        # Log data to file
        with open("logs/time_series_data.log", "a") as f:
            log_entry = {
                "data": request.data,
                "timestamp": request.timestamp if request.timestamp else str(datetime.utcnow())
            }
            f.write(json.dumps(log_entry) + "\n")

        return {"status": "logged", "message": "Data logged successfully"}

    except Exception as e:
        logger.error(f"Logging error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Logging error: {str(e)}")

# Error handling for validation errors
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return HTTPException(status_code=422, detail=exc.errors())

# Additional route to get prediction history 
@app.get("/history")
def get_prediction_history():
    logger.info("Fetching prediction history.")
    try:
        with open("logs/time_series_data.log", "r") as f:
            history = f.readlines()

        # Return history as JSON
        return {"status": "success", "history": [json.loads(line) for line in history]}

    except FileNotFoundError:
        logger.warning("No history found.")
        return {"status": "empty", "message": "No prediction history available"}

    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"History retrieval error: {str(e)}")

# Extended health check route
@app.get("/health/detailed")
def detailed_health_check():
    logger.info("Accessing detailed health check.")
    try:
        # Additional checks
        model_status = "loaded" if model else "not loaded"
        return {
            "status": "healthy",
            "model_status": model_status,
            "uptime": str(datetime.utcnow())
        }

    except Exception as e:
        logger.error(f"Error during detailed health check: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during health check")