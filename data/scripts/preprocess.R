library(zoo)         
library(dplyr)       
library(scales)     
library(caret)        
library(lubridate)    
library(ggplot2)      
library(futile.logger) 
library(tsoutliers)   

# Initialize logger
flog.appender(appender.file("logs/preprocess.log"))
flog.threshold(INFO)

# Define file paths
raw_data_path <- "data/raw/time_series_data.csv"
processed_data_path <- "data/processed/preprocessed_data.csv"
feature_data_path <- "data/features/engineered_features.csv"

# Start preprocessing
flog.info("Starting data preprocessing...")

# Load raw time-series data
data <- read.csv(raw_data_path)
flog.info("Raw data loaded from %s", raw_data_path)

# Convert date column to Date/Time format
data$date <- as.POSIXct(data$date, format="%Y-%m-%d %H:%M:%S", tz="UTC")
flog.info("Date column converted to POSIXct format")

# Sort data by date
data <- data %>% arrange(date)
flog.info("Data sorted by date")

# Handle missing values using linear interpolation
data_clean <- data %>%
  mutate(across(where(is.numeric), ~ na.approx(., na.rm = FALSE)))
flog.info("Missing values handled using linear interpolation")

# Impute remaining missing values using K-Nearest Neighbors (KNN)
preprocess_model <- preProcess(data_clean, method='knnImpute')
data_imputed <- predict(preprocess_model, data_clean)
flog.info("Remaining missing values imputed using KNN")

# Outlier detection using tsoutliers package
outliers <- tso(data_imputed[,-1]) 
data_imputed$outlier_flag <- ifelse(seq(nrow(data_imputed)) %in% outliers$index, 1, 0)
flog.info("Outliers detected and flagged in the dataset")

# Save outlier-free data separately
outlier_free_data <- data_imputed %>% filter(outlier_flag == 0)
flog.info("Outlier-free data extracted")

# Feature Engineering: Add rolling mean and standard deviation
data_features <- data_imputed %>%
  mutate(rolling_mean_7 = rollapplyr(value, width = 7, FUN = mean, fill = NA),
         rolling_sd_7 = rollapplyr(value, width = 7, FUN = sd, fill = NA),
         rolling_mean_30 = rollapplyr(value, width = 30, FUN = mean, fill = NA),
         rolling_sd_30 = rollapplyr(value, width = 30, FUN = sd, fill = NA))
flog.info("Feature engineering complete: rolling mean and SD added")

# Save engineered features
write.csv(data_features, feature_data_path, row.names = FALSE)
flog.info("Engineered features saved to %s", feature_data_path)

# Normalize numeric columns (min-max scaling)
data_normalized <- data_features %>%
  mutate(across(where(is.numeric), ~ rescale(., to = c(0, 1))))
flog.info("Numeric columns normalized using min-max scaling")

# Advanced Normalization: Z-score normalization
data_zscore <- data_features %>%
  mutate(across(where(is.numeric), ~ (.-mean(.))/sd(.)))
flog.info("Numeric columns normalized using Z-score normalization")

# Plot time-series data before and after preprocessing
ggplot(data, aes(x = date, y = value)) +
  geom_line() +
  ggtitle("Original Time-Series Data") +
  xlab("Date") + ylab("Value") +
  theme_minimal()

ggplot(data_normalized, aes(x = date, y = value)) +
  geom_line() +
  ggtitle("Normalized Time-Series Data") +
  xlab("Date") + ylab("Normalized Value") +
  theme_minimal()

flog.info("Visualization of original and normalized data complete")

# Log summary statistics of the dataset
flog.info("Summary of cleaned and imputed data:")
flog.info(summary(data_clean))
flog.info("Summary of feature-engineered data:")
flog.info(summary(data_features))

# Save preprocessed data
write.csv(data_normalized, processed_data_path, row.names = FALSE)
flog.info("Preprocessed data saved to %s", processed_data_path)

# Additional Operations for Time-Series:
# 1. Differencing to remove trends
data_diff <- data %>%
  mutate(diff_value = c(NA, diff(value)))
flog.info("Differencing applied to remove trends")

# 2. Log transformation to stabilize variance
data_log <- data %>%
  mutate(log_value = log(value + 1))  # Avoid log(0) by adding 1
flog.info("Log transformation applied")

# Seasonal Decomposition
decomposition <- decompose(ts(data_imputed$value, frequency = 365), type = "multiplicative")
plot(decomposition)
flog.info("Seasonal decomposition complete")

# Fourier Transforms (FFT) for frequency domain analysis
fft_values <- fft(data_imputed$value)
flog.info("Fourier Transform applied to time-series data")

# Inverse FFT for signal reconstruction
ifft_values <- Re(fft(fft_values, inverse = TRUE)/length(fft_values))
data_imputed$reconstructed_signal <- ifft_values
flog.info("Inverse FFT applied to reconstruct signal")

# Handling anomalies: Mark outliers in the dataset
anomalies <- tso(ts(data_imputed$value, frequency = 365))$outliers
if (length(anomalies) > 0) {
  data_imputed <- data_imputed %>% mutate(anomaly_flag = ifelse(row_number() %in% anomalies$index, 1, 0))
  flog.info("Anomalies flagged in the dataset")
} else {
  flog.info("No anomalies detected")
}

# Generate a report with key statistics
report_file <- "reports/preprocessing_report.txt"
report <- file(report_file, "w")
writeLines(c("Preprocessing Report:",
             "================================",
             paste("Rows: ", nrow(data)),
             paste("Columns: ", ncol(data)),
             "",
             "Missing Value Strategy:",
             "Linear Interpolation followed by KNN Imputation",
             "",
             "Normalization Strategy:",
             "Min-Max Scaling and Z-Score Normalization",
             "",
             "Outliers Detected: ", nrow(data_imputed %>% filter(outlier_flag == 1)),
             "Anomalies Detected: ", nrow(data_imputed %>% filter(anomaly_flag == 1))),
           con = report)
close(report)
flog.info("Preprocessing report generated at %s", report_file)

# Final log message to indicate the script is complete
flog.info("Data preprocessing complete. Check logs for details.")