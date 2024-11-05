library(testthat)
library(forecast)

# Source the model script
source("../models/r_models.R")

# Test Suite for R-based Anomaly Detection Models
test_that("SARIMA model trains and predicts without errors", {
  # Generate synthetic time-series data for testing
  ts_data <- ts(rnorm(100, mean = 10, sd = 3), frequency = 12)
  
  # Train the SARIMA model
  model <- auto.arima(ts_data)
  
  # Check if the model was created successfully
  expect_s3_class(model, "ARIMA")

  # Make predictions
  forecast_values <- forecast(model, h = 10)
  
  # Check that the forecast object is of the correct class
  expect_s3_class(forecast_values, "forecast")
  
  # Check that the forecast has the expected length
  expect_equal(length(forecast_values$mean), 10)
})

test_that("SARIMA model handles missing data", {
  # Generate time-series data with missing values
  ts_data_with_na <- ts(c(rnorm(95, mean = 10, sd = 3), rep(NA, 5)), frequency = 12)
  
  # Handle missing data by imputing (mean imputation for this test)
  ts_data_imputed <- na.interp(ts_data_with_na)
  
  # Check that there are no more missing values
  expect_true(all(!is.na(ts_data_imputed)))
  
  # Train the SARIMA model on imputed data
  model <- auto.arima(ts_data_imputed)
  
  # Check if the model was created successfully
  expect_s3_class(model, "ARIMA")
})

test_that("Model evaluation metrics are computed correctly", {
  # Generate synthetic time-series data for testing
  ts_train <- ts(rnorm(100, mean = 10, sd = 3), frequency = 12)
  ts_test <- ts(rnorm(20, mean = 10, sd = 3), frequency = 12)
  
  # Train a SARIMA model
  model <- auto.arima(ts_train)
  
  # Make predictions
  predictions <- forecast(model, h = length(ts_test))$mean
  
  # Compute evaluation metrics (e.g., RMSE)
  rmse <- sqrt(mean((predictions - ts_test)^2))
  
  # Check that the RMSE is a numeric value
  expect_true(is.numeric(rmse))
  expect_gt(rmse, 0)
})

test_that("SARIMA model raises errors for invalid input", {
  # Test with non-time-series data
  invalid_data <- c(1, 2, 3, "a", 5)
  
  # Expect an error when trying to fit a model on invalid data
  expect_error(auto.arima(invalid_data), "non-numeric argument")
})

# Additional Tests for Robustness

test_that("SARIMA model works with large datasets", {
  # Generate a large time-series dataset
  large_ts_data <- ts(rnorm(10000, mean = 20, sd = 5), frequency = 12)
  
  # Train the SARIMA model
  model <- auto.arima(large_ts_data)
  
  # Check if the model was created successfully
  expect_s3_class(model, "ARIMA")
  
  # Make predictions for a larger forecast horizon
  forecast_values <- forecast(model, h = 100)
  
  # Check that the forecast has the expected length
  expect_equal(length(forecast_values$mean), 100)
})

test_that("SARIMA model with seasonal components", {
  # Generate synthetic seasonal time-series data
  ts_seasonal <- ts(rnorm(120, mean = 15, sd = 3) + sin(2 * pi * (1:120) / 12), frequency = 12)
  
  # Train the SARIMA model with seasonal components
  model <- auto.arima(ts_seasonal)
  
  # Check if the model was created successfully
  expect_s3_class(model, "ARIMA")
  
  # Check for seasonal components in the model
  expect_true(!is.null(model$arma[5]) && model$arma[5] > 0)
  
  # Make predictions
  forecast_values <- forecast(model, h = 12)
  
  # Check that the forecast captures seasonality
  expect_true(any(diff(forecast_values$mean) != 0))
})

# Cross-Validation and Stability Tests

test_that("Cross-validation for time-series model", {
  # Generate synthetic time-series data
  ts_data_cv <- ts(rnorm(200, mean = 12, sd = 4), frequency = 12)
  
  # Perform time-series cross-validation (rolling forecast origin)
  errors <- c()
  for (i in seq(150, 190, by = 5)) {
    train <- ts_data_cv[1:i]
    test <- ts_data_cv[(i+1):(i+5)]
    
    # Train SARIMA model
    model <- auto.arima(train)
    
    # Forecast and compute error
    forecast_values <- forecast(model, h = 5)$mean
    cv_error <- sqrt(mean((forecast_values - test)^2))
    errors <- c(errors, cv_error)
  }
  
  # Check that the cross-validation errors are numeric and positive
  expect_true(all(is.numeric(errors)))
  expect_true(all(errors > 0))
})

test_that("Model stability with noisy data", {
  # Generate synthetic noisy time-series data
  ts_noisy <- ts(rnorm(100, mean = 10, sd = 10), frequency = 12)
  
  # Train the SARIMA model
  model <- auto.arima(ts_noisy)
  
  # Make predictions
  forecast_values <- forecast(model, h = 10)
  
  # Check that the model produces forecasts without errors
  expect_s3_class(forecast_values, "forecast")
  
  # Evaluate forecast uncertainty (check prediction intervals)
  expect_true(all(!is.na(forecast_values$lower)))
  expect_true(all(!is.na(forecast_values$upper)))
})

# Edge Case Tests

test_that("Model handles constant time-series data", {
  # Generate a constant time-series
  ts_constant <- ts(rep(5, 100), frequency = 12)
  
  # Train SARIMA model
  model <- auto.arima(ts_constant)
  
  # Make predictions
  forecast_values <- forecast(model, h = 10)
  
  # Check if forecast maintains the constant value
  expect_equal(mean(forecast_values$mean), 5)
})

test_that("Model handles short time-series data", {
  # Generate a very short time-series
  ts_short <- ts(rnorm(5, mean = 10, sd = 3), frequency = 12)
  
  # Expect an error or warning when trying to fit a model
  expect_warning(auto.arima(ts_short), "series too short")
})

# Performance Tests

test_that("Model performance within acceptable time limits", {
  # Generate a medium-sized time-series
  ts_perf <- ts(rnorm(5000, mean = 15, sd = 5), frequency = 12)
  
  # Record the time taken to train the model
  start_time <- Sys.time()
  model <- auto.arima(ts_perf)
  end_time <- Sys.time()
  
  training_time <- end_time - start_time
  
  # Ensure the model trains within a reasonable time (< 10 seconds)
  expect_lt(as.numeric(training_time, units = "secs"), 10)
})

test_that("Model performance on varying forecast horizons", {
  # Generate a medium-sized time-series
  ts_forecast <- ts(rnorm(500, mean = 10, sd = 4), frequency = 12)
  
  # Train SARIMA model
  model <- auto.arima(ts_forecast)
  
  # Test performance on different forecast horizons
  horizons <- c(5, 10, 20, 50)
  for (h in horizons) {
    forecast_values <- forecast(model, h = h)
    
    # Check forecast length matches the horizon
    expect_equal(length(forecast_values$mean), h)
  }
})

# Stress Testing on Anomalous Data

test_that("Model handles sudden spikes in data", {
  # Generate time-series with sudden spikes
  ts_spikes <- ts(c(rnorm(90, mean = 10, sd = 3), 100, rnorm(9, mean = 10, sd = 3)), frequency = 12)
  
  # Train SARIMA model
  model <- auto.arima(ts_spikes)
  
  # Check model handles spikes without errors
  forecast_values <- forecast(model, h = 10)
  
  # Check prediction intervals account for the spike
  expect_true(any(forecast_values$mean < 50))
})

test_that("Model handles long-term trends", {
  # Generate time-series with a long-term upward trend
  ts_trend <- ts(cumsum(rnorm(100, mean = 1, sd = 0.5)), frequency = 12)
  
  # Train SARIMA model
  model <- auto.arima(ts_trend)
  
  # Make predictions
  forecast_values <- forecast(model, h = 10)
  
  # Check if the forecast continues the upward trend
  expect_true(all(diff(forecast_values$mean) > 0))
})