library(Metrics)
library(dplyr)

# Function to calculate Precision
precision_metric <- function(true_labels, predicted_labels) {
  true_positive <- sum(true_labels == 1 & predicted_labels == 1)
  false_positive <- sum(true_labels == 0 & predicted_labels == 1)
  precision <- true_positive / (true_positive + false_positive)
  return(precision)
}

# Function to calculate Recall
recall_metric <- function(true_labels, predicted_labels) {
  true_positive <- sum(true_labels == 1 & predicted_labels == 1)
  false_negative <- sum(true_labels == 1 & predicted_labels == 0)
  recall <- true_positive / (true_positive + false_negative)
  return(recall)
}

# Function to calculate F1-Score
f1_score_metric <- function(true_labels, predicted_labels) {
  precision <- precision_metric(true_labels, predicted_labels)
  recall <- recall_metric(true_labels, predicted_labels)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  return(f1_score)
}

# Function to calculate Accuracy
accuracy_metric <- function(true_labels, predicted_labels) {
  accuracy <- sum(true_labels == predicted_labels) / length(true_labels)
  return(accuracy)
}

# Function to calculate Specificity
specificity_metric <- function(true_labels, predicted_labels) {
  true_negative <- sum(true_labels == 0 & predicted_labels == 0)
  false_positive <- sum(true_labels == 0 & predicted_labels == 1)
  specificity <- true_negative / (true_negative + false_positive)
  return(specificity)
}

# Function to calculate AUC (Area Under the Curve)
auc_metric <- function(true_labels, predicted_probs) {
  auc <- auc(true_labels, predicted_probs)
  return(auc)
}

# Function to calculate Matthews Correlation Coefficient (MCC)
mcc_metric <- function(true_labels, predicted_labels) {
  tp <- sum(true_labels == 1 & predicted_labels == 1)
  tn <- sum(true_labels == 0 & predicted_labels == 0)
  fp <- sum(true_labels == 0 & predicted_labels == 1)
  fn <- sum(true_labels == 1 & predicted_labels == 0)
  
  numerator <- (tp * tn) - (fp * fn)
  denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  
  mcc <- numerator / denominator
  return(mcc)
}

# Utility to calculate all metrics and return them in a named list
calculate_all_metrics <- function(true_labels, predicted_labels, predicted_probs) {
  metrics <- list(
    precision = precision_metric(true_labels, predicted_labels),
    recall = recall_metric(true_labels, predicted_labels),
    f1_score = f1_score_metric(true_labels, predicted_labels),
    accuracy = accuracy_metric(true_labels, predicted_labels),
    specificity = specificity_metric(true_labels, predicted_labels),
    auc = auc_metric(true_labels, predicted_probs),
    mcc = mcc_metric(true_labels, predicted_labels)
  )
  
  return(metrics)
}