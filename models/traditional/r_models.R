library(forecast)
library(tseries)
library(ggplot2)
library(caret)
library(dplyr)

# Load and inspect the dataset
data <- ts(read.csv("data/processed/time_series_data.csv")[,2], frequency = 365)
print("Initial data preview:")
print(head(data))

# Plot the raw time-series data
plot.ts(data, main = "Raw Time-Series Data", col = "blue", ylab = "Values")

# Step 1: Data Preprocessing
cat("\nStep 1: Data Preprocessing\n")

# Handle missing values - forward fill
data <- na.locf(data)
cat("Missing values handled using forward fill.\n")

# Inspect the processed data
cat("Processed Data:\n")
print(head(data))

# Differencing for stationarity
adf_test <- adf.test(data)
if (adf_test$p.value > 0.05) {
  data_diff <- diff(data)
  cat("Data differenced to achieve stationarity.\n")
} else {
  data_diff <- data
  cat("Data already stationary.\n")
}

# Plot the differenced data
plot.ts(data_diff, main = "Differenced Time-Series Data", col = "green")

# Step 2: Feature Engineering
cat("\nStep 2: Feature Engineering\n")

# Create rolling mean and standard deviation features
window_size <- 7
rolling_mean <- rollmean(data, k = window_size, fill = NA)
rolling_sd <- rollapply(data, width = window_size, FUN = sd, fill = NA)

# Create lag features
lag_1 <- lag(data, -1)
lag_7 <- lag(data, -7)

# Combine features into a data frame
feature_df <- data.frame(data, rolling_mean, rolling_sd, lag_1, lag_7)
cat("Feature engineering completed. Added rolling statistics and lag features.\n")

# Plot the rolling statistics
plot.ts(rolling_mean, col = "purple", main = "Rolling Mean Feature", ylab = "Values")
plot.ts(rolling_sd, col = "orange", main = "Rolling Standard Deviation Feature", ylab = "Values")

# Step 3: Model Development - SARIMA
cat("\nStep 3: Model Development\n")

# Automatically fit SARIMA model
sarima_model <- auto.arima(data_diff, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
cat("SARIMA model fitted.\n")
print(summary(sarima_model))

# Forecast for the next 30 days
forecast_horizon <- 30
sarima_forecast <- forecast(sarima_model, h = forecast_horizon)

# Step 4: Hyperparameter Tuning using Grid Search
cat("\nStep 4: Hyperparameter Tuning\n")

# Define grid for ARIMA orders
grid <- expand.grid(p = 0:2, d = 0:1, q = 0:2)
best_aic <- Inf
best_model <- NULL

for (i in 1:nrow(grid)) {
  model <- try(arima(data_diff, order = c(grid$p[i], grid$d[i], grid$q[i]), seasonal = list(order = c(1, 1, 1), period = 365)), silent = TRUE)
  if (inherits(model, "try-error")) next
  if (AIC(model) < best_aic) {
    best_aic <- AIC(model)
    best_model <- model
  }
}

cat("Best SARIMA model selected with AIC:", best_aic, "\n")
print(best_model)

# Step 5: Cross-Validation
cat("\nStep 5: Cross-Validation\n")

# Define time-series cross-validation function
ts_cv <- function(data, h) {
  n <- length(data)
  errors <- numeric(n - h)
  for (i in seq_len(n - h)) {
    train_data <- data[1:i]
    test_data <- data[(i + 1):(i + h)]
    model <- auto.arima(train_data)
    forecasted <- forecast(model, h = h)
    errors[i] <- mean(abs(forecasted$mean - test_data) / test_data)
  }
  return(mean(errors))
}

# Perform 5-fold cross-validation
cv_error <- ts_cv(data_diff, h = 5)
cat("Cross-validation error (MAPE):", cv_error, "\n")

# Step 6: Model Diagnostics
cat("\nStep 6: Model Diagnostics\n")

# Residual diagnostics
checkresiduals(sarima_model)

# Plot residuals
residuals <- residuals(sarima_model)
plot.ts(residuals, main = "Residuals from SARIMA Model", col = "red")

# Step 7: Anomaly Detection
cat("\nStep 7: Anomaly Detection\n")

# Define threshold for anomalies (2 standard deviations)
threshold <- 2 * sd(residuals)

# Detect anomalies
anomalies <- which(abs(residuals) > threshold)
cat("Anomalies detected at time points:", anomalies, "\n")

# Step 8: Plot Results
cat("\nStep 8: Plot Results\n")

# Plot the forecast with anomalies
plot(forecast(sarima_model, h = forecast_horizon), main = "SARIMA Forecast with Anomalies")
points(anomalies, data[anomalies], col = "red", pch = 19, cex = 1.5)

# Step 9: Save Model and Results
cat("\nStep 9: Save Model and Results\n")

# Save the SARIMA model
saveRDS(sarima_model, file = "models/traditional/sarima_model.rds")
cat("SARIMA model saved.\n")

# Save anomalies to a CSV
write.csv(anomalies, "results/anomalies.csv")
cat("Anomalies saved to 'results/anomalies.csv'.\n")

# Step 10: Generate Detailed Anomaly Report
cat("\nStep 10: Generate Detailed Anomaly Report\n")

# Create a report of anomalies with associated residuals
anomaly_report <- data.frame(Time = anomalies, Value = data[anomalies], Residual = residuals[anomalies])
write.csv(anomaly_report, "results/anomaly_report.csv")
cat("Anomaly report saved to 'results/anomaly_report.csv'.\n")

# Step 11: Evaluation Metrics
cat("\nStep 11: Evaluation Metrics\n")

# Compute Mean Absolute Percentage Error (MAPE)
mape <- mean(abs(residuals) / data) * 100
cat("Model MAPE: ", mape, "%\n")

# Step 12: Additional Visualization
cat("\nStep 12: Additional Visualization\n")

# Create a ggplot visualization of the residuals and anomalies
ggplot(data = data.frame(Time = 1:length(data), Value = data, Residuals = residuals)) +
  geom_line(aes(x = Time, y = Residuals), color = "blue") +
  geom_point(aes(x = anomalies, y = residuals[anomalies]), color = "red", size = 3) +
  ggtitle("Residuals and Detected Anomalies") +
  theme_minimal()

# Step 13: Export Model Summary and Forecast
cat("\nStep 13: Export Model Summary and Forecast\n")

# Save forecast to CSV
forecast_df <- data.frame(Time = time(sarima_forecast$mean), Forecast = sarima_forecast$mean, Lower = sarima_forecast$lower[, 2], Upper = sarima_forecast$upper[, 2])
write.csv(forecast_df, "results/forecast.csv")
cat("Forecast saved to 'results/forecast.csv'.\n")

cat("\nScript Execution Completed Successfully.\n")