library(ggplot2)
library(dplyr)
library(futile.logger)
library(lubridate)
library(scales)

# Set up logger
flog.appender(appender.file("monitoring_log.txt"))
flog.threshold(INFO)

# Helper function to load the dataset
load_metrics <- function(file_path) {
  flog.info("Loading metrics data from file: %s", file_path)
  if (!file.exists(file_path)) {
    flog.error("File %s does not exist.", file_path)
    stop("File not found.")
  }
  
  metrics_data <- tryCatch({
    read.csv(file_path, stringsAsFactors = FALSE)
  }, error = function(e) {
    flog.error("Error in reading the file: %s", e)
    stop("Error in reading CSV file.")
  })
  
  if ("date" %in% colnames(metrics_data)) {
    metrics_data$date <- ymd_hms(metrics_data$date)
  } else {
    flog.error("Date column is missing from the metrics data.")
    stop("Invalid data format.")
  }
  
  flog.info("Metrics data successfully loaded with %d rows.", nrow(metrics_data))
  return(metrics_data)
}

# Calculate precision, recall, F1-score, and accuracy
calculate_performance <- function(metrics_data) {
  flog.info("Calculating precision, recall, F1-score, and accuracy.")
  
  performance_summary <- metrics_data %>%
    group_by(date) %>%
    summarize(
      true_positive = sum(true_positive, na.rm = TRUE),
      false_positive = sum(false_positive, na.rm = TRUE),
      false_negative = sum(false_negative, na.rm = TRUE),
      true_negative = sum(true_negative, na.rm = TRUE),
      precision = sum(true_positive) / (sum(true_positive) + sum(false_positive)),
      recall = sum(true_positive) / (sum(true_positive) + sum(false_negative)),
      accuracy = (sum(true_positive) + sum(true_negative)) / (sum(true_positive) + sum(false_positive) + sum(false_negative) + sum(true_negative)),
      f1_score = 2 * (precision * recall) / (precision + recall)
    )
  
  flog.info("Performance metrics successfully calculated.")
  return(performance_summary)
}

# Generate performance plots with enhanced visualization
plot_performance <- function(performance_summary) {
  flog.info("Generating performance plots.")
  
  precision_plot <- ggplot(performance_summary, aes(x = date, y = precision)) +
    geom_line(color = "blue", size = 1.2) +
    ggtitle("Precision Over Time") +
    xlab("Date") + ylab("Precision") +
    scale_x_datetime(labels = date_format("%Y-%m-%d"), breaks = "1 month") +
    theme_minimal()
  
  recall_plot <- ggplot(performance_summary, aes(x = date, y = recall)) +
    geom_line(color = "green", size = 1.2) +
    ggtitle("Recall Over Time") +
    xlab("Date") + ylab("Recall") +
    scale_x_datetime(labels = date_format("%Y-%m-%d"), breaks = "1 month") +
    theme_minimal()
  
  f1_score_plot <- ggplot(performance_summary, aes(x = date, y = f1_score)) +
    geom_line(color = "red", size = 1.2) +
    ggtitle("F1 Score Over Time") +
    xlab("Date") + ylab("F1 Score") +
    scale_x_datetime(labels = date_format("%Y-%m-%d"), breaks = "1 month") +
    theme_minimal()
  
  accuracy_plot <- ggplot(performance_summary, aes(x = date, y = accuracy)) +
    geom_line(color = "purple", size = 1.2) +
    ggtitle("Accuracy Over Time") +
    xlab("Date") + ylab("Accuracy") +
    scale_x_datetime(labels = date_format("%Y-%m-%d"), breaks = "1 month") +
    theme_minimal()
  
  flog.info("Performance plots generated.")
  return(list(precision_plot = precision_plot, recall_plot = recall_plot, f1_score_plot = f1_score_plot, accuracy_plot = accuracy_plot))
}

# Enhanced function to generate and save detailed visualizations
save_plots <- function(plots) {
  flog.info("Saving plots to files.")
  
  tryCatch({
    ggsave("precision_over_time.png", plot = plots$precision_plot, width = 10, height = 6)
    ggsave("recall_over_time.png", plot = plots$recall_plot, width = 10, height = 6)
    ggsave("f1_score_over_time.png", plot = plots$f1_score_plot, width = 10, height = 6)
    ggsave("accuracy_over_time.png", plot = plots$accuracy_plot, width = 10, height = 6)
  }, error = function(e) {
    flog.error("Error saving plots: %s", e)
    stop("Failed to save plots.")
  })
  
  flog.info("All plots saved successfully.")
}

# Helper function to print summary statistics
print_summary <- function(performance_summary) {
  flog.info("Printing summary statistics for the latest date.")
  
  latest_data <- tail(performance_summary, 1)
  
  cat("Summary for", latest_data$date, "\n")
  cat("Precision:", round(latest_data$precision, 4), "\n")
  cat("Recall:", round(latest_data$recall, 4), "\n")
  cat("F1 Score:", round(latest_data$f1_score, 4), "\n")
  cat("Accuracy:", round(latest_data$accuracy, 4), "\n")
  
  flog.info("Summary statistics printed.")
}

# Monitoring and performance analysis main function
monitor_metrics <- function(metrics_file) {
  flog.info("Starting monitoring process for file: %s", metrics_file)
  
  # Load metrics data
  metrics_data <- load_metrics(metrics_file)
  
  # Calculate performance metrics
  performance_summary <- calculate_performance(metrics_data)
  
  # Generate performance plots
  plots <- plot_performance(performance_summary)
  
  # Save plots
  save_plots(plots)
  
  # Print summary statistics
  print_summary(performance_summary)
  
  flog.info("Monitoring process completed.")
}

# Execute monitoring with specified metrics file
metrics_file <- "metrics_log.csv" 
flog.info("Executing monitor.R script.")
monitor_metrics(metrics_file)

# Final log and process completion
flog.info("Script execution completed successfully.")