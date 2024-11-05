# Load necessary libraries
library(plumber)
library(jsonlite)
library(futile.logger)

# Initialize logger
flog.threshold(INFO)
flog.appender(appender.file("logs/api.log"))

#* @apiTitle Time-Series Anomaly Detection API
#* @apiDescription API to serve an anomaly detection model for time-series data.
#* @apiVersion 1.0

# Load the pre-trained R model (adjust the path to the actual model location)
load_model <- function() {
  flog.info("Loading model...")
  model <- tryCatch({
    readRDS("models/r_trained_model.rds")
  }, error = function(e) {
    flog.error("Failed to load model: %s", e$message)
    stop("Model loading failed.")
  })
  flog.info("Model successfully loaded.")
  return(model)
}

# Validate input data structure
validate_input <- function(input_data) {
  flog.info("Validating input data structure...")
  if (!is.data.frame(input_data)) {
    flog.error("Input data is not a valid dataframe.")
    stop("Invalid input: Expected a dataframe.")
  }
  if (ncol(input_data) < 1) {
    flog.error("Input dataframe has no columns.")
    stop("Invalid input: Dataframe has no columns.")
  }
  flog.info("Input data is valid.")
}

# Predict anomalies using the model
predict_anomaly <- function(input_data, model) {
  flog.info("Predicting anomalies...")
  
  # Checking for required columns
  if (!("timestamp" %in% colnames(input_data))) {
    flog.error("Input data does not contain 'timestamp' column.")
    stop("Invalid input: Missing 'timestamp' column.")
  }
  
  prediction <- tryCatch({
    predict(model, newdata = input_data)
  }, error = function(e) {
    flog.error("Prediction failed: %s", e$message)
    stop("Prediction failed.")
  })
  
  flog.info("Anomaly prediction successful.")
  return(prediction)
}

#* Health check endpoint
#* @get /health
#* @response 200 Returns "API is running."
function() {
  flog.info("Health check triggered.")
  list(status = "API is running.")
}

#* Model info endpoint
#* @get /model_info
#* @response 200 Returns model details.
function() {
  flog.info("Model info request triggered.")
  
  model <- load_model()
  
  model_info <- list(
    type = class(model),
    trained_on = "time-series data",
    prediction_type = "anomalies"
  )
  
  return(model_info)
}

# Log the request and response
log_request_response <- function(req, res) {
  flog.info("Request received: %s", req$PATH_INFO)
  flog.info("Response status: %s", res$status)
}

#* Anomaly detection endpoint
#* @post /detect
#* @param data:json The input time-series data in JSON format.
#* @response 200 Returns anomaly detection results.
function(req, res) {
  flog.info("Anomaly detection request triggered.")
  
  # Parse input data
  input_data <- tryCatch({
    fromJSON(req$postBody)
  }, error = function(e) {
    flog.error("Failed to parse input data: %s", e$message)
    res$status <- 400
    return(list(error = "Invalid input data format."))
  })
  
  # Validate input data
  validate_input(input_data)
  
  # Load the model
  model <- load_model()

  # Predict anomalies
  results <- predict_anomaly(input_data, model)
  
  # Log the response
  log_request_response(req, res)
  
  # Return results
  res$status <- 200
  return(list(predictions = results))
}

#* Sample input data endpoint
#* @get /sample_input
#* @response 200 Returns a sample input format for anomaly detection.
function() {
  flog.info("Sample input request triggered.")
  
  sample_data <- data.frame(
    timestamp = as.POSIXct(c('2024-01-01 00:00:00', '2024-01-01 00:01:00')),
    value = c(10.5, 11.0)
  )
  
  return(toJSON(sample_data, pretty = TRUE))
}

#* Logging endpoint
#* @get /logs
#* @response 200 Returns the latest logs.
function() {
  flog.info("Logs request triggered.")
  
  log_data <- tryCatch({
    readLines("logs/api.log")
  }, error = function(e) {
    flog.error("Failed to read logs: %s", e$message)
    stop("Log reading failed.")
  })
  
  return(list(logs = log_data))
}

# Error handler for the API
#* @plumber error
function(req, res, err) {
  flog.error("Error encountered: %s", err$message)
  res$status <- 500
  list(error = err$message)
}

# Create plumber API router
api <- plumber$new()

# Add CORS headers
api$addGlobalProcessor(plumber::cors())

# Add error logging middleware
api$setErrorHandler(function(req, res, err) {
  flog.error("Global error handler: %s", err$message)
  res$status <- 500
  list(error = err$message)
})

# API documentation
api$handle("GET", "/docs", function() {
  paste("This API serves a time-series anomaly detection model. Endpoints include:",
        "/health - Health check",
        "/model_info - Information about the loaded model",
        "/detect - Anomaly detection on time-series data",
        "/sample_input - Get a sample input format",
        "/logs - View the latest logs",
        sep = "\n")
})

#* @get /about
#* @response 200 Returns information about the API and its usage.
function() {
  flog.info("About endpoint accessed.")
  
  info <- list(
    name = "Time-Series Anomaly Detection API",
    version = "1.0",
    description = "This API allows users to detect anomalies in time-series data using a pre-trained model.",
    endpoints = list(
      health = "/health",
      model_info = "/model_info",
      detect = "/detect",
      sample_input = "/sample_input",
      logs = "/logs",
      about = "/about"
    )
  )
  
  return(info)
}

# Start the API
flog.info("Starting API on port 8000...")
api$run(host = "0.0.0.0", port = 8000)