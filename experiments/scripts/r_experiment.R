library(caret)
library(forecast)
library(tseries)
library(ggplot2)
library(lubridate)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Load the time-series data
data <- read.csv("data/processed/time_series_data.csv")

# Preprocessing the time-series data
data$timestamp <- ymd_hms(data$timestamp)
data <- data %>% arrange(timestamp)

# Convert to time-series object with daily frequency
ts_data <- ts(data$value, frequency = 365)

# Initial summary statistics of the data
summary(data)
print(paste("Number of missing values: ", sum(is.na(data$value))))
print(paste("Data starts from: ", min(data$timestamp)))
print(paste("Data ends at: ", max(data$timestamp)))

# Visualize the original time-series data
ggplot(data, aes(x = timestamp, y = value)) +
  geom_line(color = 'blue') +
  ggtitle('Original Time-Series Data') +
  xlab('Timestamp') +
  ylab('Value')

# Decompose time-series into trend, seasonal, and random components
decomp <- stl(ts_data, s.window = "periodic")
autoplot(decomp) +
  ggtitle('Decomposition of Time-Series') +
  xlab('Time') +
  ylab('Value')

# Define the SARIMA model with caret for hyperparameter tuning
fitControl <- trainControl(method = "cv", number = 5, savePredictions = TRUE)

# Hyperparameter grid for SARIMA
sarimaGrid <- expand.grid(p = 0:3, d = 0:2, q = 0:3, P = 0:2, D = 0:1, Q = 0:2, period = c(12))

# Train the SARIMA model
sarimaModel <- train(
  ts_data ~ ., 
  method = "Arima",
  tuneGrid = sarimaGrid,
  trControl = fitControl
)

# Print the best model parameters
print(sarimaModel$bestTune)

# Save the trained model
saveRDS(sarimaModel, "models/sarima_model.rds")

# Perform anomaly detection using the best SARIMA model
best_model <- Arima(ts_data, 
                    order = c(sarimaModel$bestTune$p, sarimaModel$bestTune$d, sarimaModel$bestTune$q),
                    seasonal = list(order = c(sarimaModel$bestTune$P, sarimaModel$bestTune$D, sarimaModel$bestTune$Q), 
                                    period = sarimaModel$bestTune$period))

# Residual diagnostics to assess the model's fit
checkresiduals(best_model)

# Calculate residuals and detect anomalies (3-sigma rule)
residuals <- residuals(best_model)
threshold <- 3 * sd(residuals)
anomalies <- residuals[abs(residuals) > threshold]

# Log the anomalies and threshold used
cat("Anomaly Detection Results:\n")
cat("Threshold for anomalies: ", threshold, "\n")
cat("Number of anomalies detected: ", length(anomalies), "\n")

# Plot residuals with anomalies marked
ggplot(data = data.frame(time = time(residuals), residuals = residuals), aes(x = time, y = residuals)) +
  geom_line(color = 'black') +
  geom_point(aes(x = time[abs(residuals) > threshold], 
                 y = residuals[abs(residuals) > threshold]), color = 'red') +
  ggtitle('Residuals with Anomalies') +
  xlab('Time') +
  ylab('Residuals')

# Plot time-series data with anomalies highlighted
autoplot(ts_data) +
  geom_point(aes(x = time(ts_data)[residuals > threshold], 
                 y = ts_data[residuals > threshold]), color = 'red') +
  ggtitle('Anomaly Detection on Time-Series Data') +
  xlab('Time') +
  ylab('Value')

# Save the anomalies detected
write.csv(anomalies, "results/anomalies.csv")

# Advanced model diagnostics - ACF and PACF plots of residuals
acf_res <- Acf(residuals)
pacf_res <- Pacf(residuals)

autoplot(acf_res) + ggtitle('ACF of Residuals')
autoplot(pacf_res) + ggtitle('PACF of Residuals')

# Perform Ljung-Box test for autocorrelation in residuals
ljung_box_test <- Box.test(residuals, lag = 10, type = "Ljung-Box")
cat("Ljung-Box Test p-value: ", ljung_box_test$p.value, "\n")

# Perform cross-validation to evaluate the SARIMA model
cat("Cross-Validation Results:\n")
print(sarimaModel$results)

# Perform rolling window forecast
n_forecast <- 30
future_forecast <- forecast(best_model, h = n_forecast)

# Plot the forecasted results
autoplot(future_forecast) +
  ggtitle(paste('Forecasting next', n_forecast, 'days')) +
  xlab('Time') +
  ylab('Forecasted Value')

# Save the forecast results
write.csv(future_forecast, "results/forecast.csv")

# Additional checks for model residuals
shapiro_test <- shapiro.test(residuals)
cat("Shapiro-Wilk normality test p-value: ", shapiro_test$p.value, "\n")

# Log-Transform the data if residuals are not normally distributed
if (shapiro_test$p.value < 0.05) {
  log_ts_data <- log(ts_data)
  best_model_log <- Arima(log_ts_data, 
                          order = c(sarimaModel$bestTune$p, sarimaModel$bestTune$d, sarimaModel$bestTune$q),
                          seasonal = list(order = c(sarimaModel$bestTune$P, sarimaModel$bestTune$D, sarimaModel$bestTune$Q), 
                                          period = sarimaModel$bestTune$period))
  residuals_log <- residuals(best_model_log)
  cat("Re-calculated residuals after log transformation.\n")
}

# Plot log-transformed time-series data
if (shapiro_test$p.value < 0.05) {
  autoplot(log_ts_data) +
    ggtitle('Log-Transformed Time-Series Data') +
    xlab('Time') +
    ylab('Log(Value)')
}

# Export model diagnostics report
diagnostics_report <- list(
  best_model_params = sarimaModel$bestTune,
  residuals_acf = acf_res,
  residuals_pacf = pacf_res,
  ljung_box_p_value = ljung_box_test$p.value,
  shapiro_wilk_p_value = shapiro_test$p.value
)

saveRDS(diagnostics_report, "results/model_diagnostics.rds")

# Create a PDF report with all plots and findings
pdf("results/model_diagnostics_report.pdf")
plot(acf_res)
plot(pacf_res)
plot(future_forecast)
plot(decomp)
dev.off()

# End of script, confirming execution finished
cat("Experiment completed and results saved to 'results/' directory.\n")