library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(reshape2)
library(PRROC)
library(rlang)  # For the .data pronoun

# Function to calculate performance metrics for binary classification
evaluate_model <- function(true_labels, predicted_labels, predicted_probs) {
  # Ensure that the factors have the same levels
  true_factor <- factor(true_labels, levels = c(0, 1))
  pred_factor <- factor(predicted_labels, levels = c(0, 1))
  
  confusion_matrix <- caret::confusionMatrix(pred_factor, true_factor, positive = "1")
  
  # Calculate AUC-ROC
  roc_curve <- pROC::roc(true_labels, predicted_probs)
  auc_value <- pROC::auc(roc_curve)
  
  # Calculate AUC-PR (Precision-Recall)
  pr_curve <- PRROC::pr.curve(scores.class0 = predicted_probs[true_labels == 0],
                             scores.class1 = predicted_probs[true_labels == 1],
                             curve = TRUE)
  auc_pr_value <- pr_curve$auc.integral
  
  # Create a list of metrics
  metrics <- list(
    accuracy = confusion_matrix$overall["Accuracy"],
    precision = confusion_matrix$byClass["Pos Pred Value"],
    recall = confusion_matrix$byClass["Sensitivity"],
    f1_score = confusion_matrix$byClass["F1"],
    auc = auc_value,
    auc_pr = auc_pr_value
  )
  
  return(metrics)
}

# Function to plot the confusion matrix as a heatmap
plot_confusion_matrix <- function(confusion_matrix) {
  cm_matrix <- confusion_matrix$table
  cm_melted <- reshape2::melt(cm_matrix)
  
  # Rename the columns to 'Prediction', 'Reference', and 'value'
  colnames(cm_melted) <- c("Prediction", "Reference", "value")
  
  plot <- ggplot(cm_melted, aes(x = .data$Reference, y = .data$Prediction, fill = .data$value)) +
    geom_tile(color = "grey") +
    geom_text(aes(label = .data$value), color = "black", size = 5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = "Confusion Matrix", x = "True Label", y = "Predicted Label") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  return(plot)
}

# Function to visualize ROC curve
plot_roc_curve <- function(true_labels, predicted_probs) {
  roc_curve <- pROC::roc(true_labels, predicted_probs)
  
  # Calculate False Positive Rate (FPR)
  fpr <- 1 - roc_curve$specificities
  tpr <- roc_curve$sensitivities
  
  roc_df <- data.frame(FPR = fpr, TPR = tpr)
  
  plot <- ggplot(roc_df, aes(x = .data$FPR, y = .data$TPR)) +
    geom_line(color = "blue", size = 1) +
    geom_abline(linetype = "dashed", color = "red") +
    labs(title = "ROC Curve", x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR)") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  return(plot)
}

# Function to plot Precision-Recall curve
plot_pr_curve <- function(true_labels, predicted_probs) {
  pr_curve <- PRROC::pr.curve(scores.class0 = predicted_probs[true_labels == 0],
                             scores.class1 = predicted_probs[true_labels == 1],
                             curve = TRUE)
  
  pr_df <- data.frame(Recall = pr_curve$curve[,1], Precision = pr_curve$curve[,2])
  
  plot <- ggplot(pr_df, aes(x = .data$Recall, y = .data$Precision)) +
    geom_line(color = "darkgreen", size = 1) +
    labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  return(plot)
}

# Function to plot model performance metrics
plot_performance_metrics <- function(metrics) {
  metric_df <- data.frame(
    metric = c("Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "AUC-PR"),
    value = unlist(metrics)
  )
  
  plot <- ggplot(metric_df, aes(x = reorder(.data$metric, -.data$value), y = .data$value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label = round(.data$value, 3)), vjust = -0.5, size = 5) +
    labs(title = "Model Performance Metrics", x = "Metric", y = "Value") +
    ylim(0, 1) +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  return(plot)
}

# Function to visualize anomaly scores distribution
plot_anomaly_distribution <- function(anomaly_scores) {
  plot <- ggplot(data.frame(scores = anomaly_scores), aes(x = .data$scores)) +
    geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
    labs(title = "Anomaly Scores Distribution", x = "Anomaly Score", y = "Frequency") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
  
  return(plot)
}

# Function to evaluate multiple models
evaluate_multiple_models <- function(models, true_labels, predicted_probs_list) {
  metrics_list <- list()
  
  for (i in seq_along(models)) {
    predicted_labels <- ifelse(predicted_probs_list[[i]] > 0.5, 1, 0)
    metrics <- evaluate_model(true_labels, predicted_labels, predicted_probs_list[[i]])
    metrics_list[[models[i]]] <- metrics
  }
  
  return(metrics_list)
}

# Cross-validation function for model evaluation
cross_validation <- function(data, model_function, k_folds = 5) {
  folds <- caret::createFolds(data$label, k = k_folds, returnTrain = TRUE)
  metrics_list <- list()
  
  for (i in seq_along(folds)) {
    train_data <- data[folds[[i]], ]
    test_data <- data[-folds[[i]], ]
    
    # Train and evaluate the model
    model <- model_function(train_data)
    predicted_probs <- predict(model, newdata = test_data, type = "prob")[,2]
    predicted_labels <- ifelse(predicted_probs > 0.5, 1, 0)
    metrics <- evaluate_model(test_data$label, predicted_labels, predicted_probs)
    
    metrics_list[[paste0("Fold_", i)]] <- metrics
  }
  
  return(metrics_list)
}

# Load dataset 
true_labels <- c(1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1)  
predicted_probs <- list(
  "Model 1" = c(0.95, 0.4, 0.3, 0.8, 0.1, 0.2, 0.9, 0.7, 0.3, 0.2, 0.85, 0.6),
  "Model 2" = c(0.92, 0.35, 0.32, 0.78, 0.15, 0.25, 0.88, 0.75, 0.35, 0.18, 0.82, 0.65)
)

models <- c("Model 1", "Model 2")

# Evaluate multiple models
metrics_list <- evaluate_multiple_models(models, true_labels, predicted_probs)

# Display evaluation metrics for all models
print(metrics_list)

# Generate visualizations for the first model
roc_plot <- plot_roc_curve(true_labels, predicted_probs[["Model 1"]])
pr_plot <- plot_pr_curve(true_labels, predicted_probs[["Model 1"]])
performance_plot <- plot_performance_metrics(metrics_list[["Model 1"]])

# Display the plots
print(roc_plot)
print(pr_plot)
print(performance_plot)

# Generate confusion matrix for the first model
predicted_labels_model1 <- ifelse(predicted_probs[["Model 1"]] > 0.5, 1, 0)
conf_matrix <- caret::confusionMatrix(factor(predicted_labels_model1, levels = c(0,1)),
                                     factor(true_labels, levels = c(0,1)),
                                     positive = "1")
conf_matrix_plot <- plot_confusion_matrix(conf_matrix)
print(conf_matrix_plot)

# Anomaly scores distribution for the first model
anomaly_distribution_plot <- plot_anomaly_distribution(predicted_probs[["Model 1"]])
print(anomaly_distribution_plot)

# Example of Cross-validation for a single model
# data <- data.frame(label = true_labels, 
#                    feature1 = runif(length(true_labels)), 
#                    feature2 = runif(length(true_labels)))
# metrics_cv <- cross_validation(data, some_model_function, k_folds = 5)