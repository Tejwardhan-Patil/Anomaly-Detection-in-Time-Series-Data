library(ggplot2)
library(dplyr)
library(forecast)
library(knitr)
library(rmarkdown)

# Set file paths
evaluation_results <- "data/processed/evaluation_results.csv"
report_output <- "reports/evaluation_report.html"

# Load evaluation results
data <- read.csv(evaluation_results)

# Generate basic statistics
stats <- data %>%
  summarize(
    Mean_Error = mean(error),
    Median_Error = median(error),
    Std_Dev = sd(error),
    Min_Error = min(error),
    Max_Error = max(error)
  )

# Create statistical plots
error_distribution <- ggplot(data, aes(x = error)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Error Distribution", x = "Error", y = "Frequency") +
  theme_minimal()

time_series_plot <- ggplot(data, aes(x = timestamp, y = error)) +
  geom_line(color = "red") +
  labs(title = "Error Over Time", x = "Time", y = "Error") +
  theme_minimal()

# Forecast future error
model <- auto.arima(data$error)
forecasted <- forecast(model, h = 10)

# Plot forecasted errors
forecast_plot <- autoplot(forecasted) +
  labs(title = "Error Forecast", x = "Time", y = "Forecasted Error") +
  theme_minimal()

# Create a report in RMarkdown
rmarkdown::render(input = "scripts/report_template.Rmd", 
                  output_file = report_output,
                  params = list(
                    stats = stats,
                    error_distribution = error_distribution,
                    time_series_plot = time_series_plot,
                    forecast_plot = forecast_plot
                  ))

# Print message on completion
message("Report generated successfully at: ", report_output)