import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import uvicorn
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

# Initialize the FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Cache
@lru_cache(maxsize=5)
def load_model(model_name: str):
    try:
        model_path = f"/{model_name}.pkl" 
        model = joblib.load(model_path)
        logger.info(f"Model {model_name} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model {model_name} could not be loaded.")

# Input schema for time-series data
class TimeSeriesRequest(BaseModel):
    data: List[float] = Field(..., title="Time-Series Data", description="A list of time-series data points")
    model_name: Optional[str] = Field("default_model", title="Model Name", description="The name of the model to use for prediction")

    @validator('data')
    def check_data_length(cls, data):
        if len(data) < 10:
            raise ValueError("Time-series data must contain at least 10 points.")
        return data

# Response schema for anomaly detection
class PredictionResponse(BaseModel):
    anomalies: List[int] = Field(..., title="Anomaly Indicators", description="List of 1 for anomaly and 0 for normal.")
    anomaly_scores: List[float] = Field(..., title="Anomaly Scores", description="Anomaly score for each data point.")

# Health check endpoint
@app.get("/", tags=["Health Check"])
def health_check():
    """Health check endpoint to ensure API is running"""
    return {"status": "API is healthy"}

# Metadata endpoint
@app.get("/model_metadata", tags=["Model Info"])
def get_model_metadata(model_name: Optional[str] = "default_model"):
    """Get metadata about the loaded model."""
    try:
        model = load_model(model_name)
        metadata = model.get_params() if hasattr(model, 'get_params') else {"info": "Metadata not available"}
        return {"model_name": model_name, "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve model metadata: {str(e)}")

# Anomaly detection endpoint
@app.post("/detect", response_model=PredictionResponse, tags=["Anomaly Detection"])
def detect_anomalies(request: TimeSeriesRequest):
    """Detect anomalies in the provided time-series data using the specified model."""
    logger.info(f"Received request to detect anomalies with model: {request.model_name}")
    
    # Convert input data to the required format
    data = np.array(request.data).reshape(-1, 1)
    df = pd.DataFrame(data, columns=["value"])

    try:
        model = load_model(request.model_name)
        logger.info("Model loaded, making predictions...")
        
        # Use the model to predict anomaly scores and anomalies
        anomaly_scores = model.decision_function(df) 
        anomalies = model.predict(df) 

        logger.info("Prediction completed successfully.")
        
        return PredictionResponse(
            anomalies=anomalies.tolist(),
            anomaly_scores=anomaly_scores.tolist()
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# Additional endpoint for model listing
@app.get("/models", tags=["Model Management"])
def list_available_models():
    """List all available models for anomaly detection."""
    available_models = ["default_model", "model_1", "model_2"] 
    return {"models": available_models}

# Validation for time-series data
def validate_time_series_data(data: List[float]):
    if not isinstance(data, list):
        logger.error("Invalid data format. Expected a list.")
        raise HTTPException(status_code=422, detail="Invalid data format. Expected a list.")
    
    if len(data) < 10:
        logger.error("Time-series data must contain at least 10 points.")
        raise HTTPException(status_code=422, detail="Time-series data must contain at least 10 points.")
    
    if any(not isinstance(x, (int, float)) for x in data):
        logger.error("All values in the time-series must be numeric.")
        raise HTTPException(status_code=422, detail="All values in the time-series must be numeric.")

    logger.info("Time-series data validated successfully.")
    return np.array(data).reshape(-1, 1)

# Error handler for bad requests
@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    logger.error(f"HTTP exception occurred: {exc.detail}")
    return {"message": exc.detail}

# Model caching endpoint for improved performance
@app.get("/cache_status", tags=["Cache Management"])
def cache_status():
    """Get information about model cache usage."""
    return {
        "cache_info": load_model.cache_info()
    }

# Inference function for serving predictions
def model_inference(model, data):
    try:
        logger.info("Running inference on the model...")
        scores = model.decision_function(data) 
        anomalies = model.predict(data) 
        return scores, anomalies
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Model inference failed")

# Extra endpoint for clearing the cache
@app.get("/clear_cache", tags=["Cache Management"])
def clear_cache():
    """Clear the model cache to free up memory."""
    load_model.cache_clear()
    logger.info("Model cache cleared.")
    return {"status": "Cache cleared"}

# Documentation with FastAPI OpenAPI schema
@app.get("/docs", tags=["Documentation"])
def get_api_documentation():
    """Retrieve the OpenAPI documentation for this API."""
    return app.openapi()

# Start the API server
if __name__ == "__main__":
    logger.info("Starting the Anomaly Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)