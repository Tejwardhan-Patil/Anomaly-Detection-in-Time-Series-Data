library(ggplot2)
library(dplyr)

# Function to plot time-series data with anomalies highlighted
plot_time_series_anomalies <- function(data, time_col, value_col, anomaly_col, title="Time Series with Anomalies") {
  
  # Convert columns to symbols for dynamic referencing
  time_sym <- sym(time_col)
  value_sym <- sym(value_col)
  anomaly_sym <- sym(anomaly_col)

  # Create a ggplot object
  p <- ggplot(data, aes(x = !!time_sym, y = !!value_sym)) +
    geom_line(color = "blue", size = 1) +
    geom_point(data = filter(data, !!anomaly_sym == TRUE), 
               aes(x = !!time_sym, y = !!value_sym), 
               color = "red", size = 3) +
    labs(title = title, x = "Time", y = "Value") +
    theme_minimal()
  
  return(p)
}

# Function to plot rolling statistics (mean, standard deviation)
plot_rolling_stats <- function(data, time_col, value_col, window_size = 10, title="Rolling Mean and Standard Deviation") {
  
  # Calculate rolling mean and standard deviation
  data <- data %>%
    arrange(!!sym(time_col)) %>%
    mutate(rolling_mean = zoo::rollmean(!!sym(value_col), k = window_size, fill = NA),
           rolling_sd = zoo::rollapply(!!sym(value_col), width = window_size, FUN = sd, fill = NA))
  
  time_sym <- sym(time_col)
  
  # Create ggplot object for rolling mean and SD
  p <- ggplot(data, aes(x = !!time_sym)) +
    geom_line(aes(y = rolling_mean), color = "green", size = 1, linetype = "solid") +
    geom_ribbon(aes(ymin = rolling_mean - rolling_sd, ymax = rolling_mean + rolling_sd), alpha = 0.3) +
    labs(title = title, x = "Time", y = "Value") +
    theme_minimal()
  
  return(p)
}

# Function to visualize feature importance (for models that support feature importance)
plot_feature_importance <- function(importance_df, feature_col, importance_col, title = "Feature Importance") {
  
  p <- ggplot(importance_df, aes(x = reorder(!!sym(feature_col), !!sym(importance_col)), y = !!sym(importance_col))) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = title, x = "Features", y = "Importance") +
    theme_minimal()
  
  return(p)
}