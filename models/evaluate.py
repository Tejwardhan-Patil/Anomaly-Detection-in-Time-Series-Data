import imblearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

def evaluate_model(model, X_test, y_test, threshold=0.5, average='binary'):
    """
    Evaluate the given model using precision, recall, F1-score, confusion matrix, and AUC-ROC.
    
    Parameters:
    - model: Trained anomaly detection model.
    - X_test: Test data (time-series).
    - y_test: True labels (0 for normal, 1 for anomaly).
    - threshold: Decision threshold for anomaly scores.
    - average: Averaging method for multi-class or multi-label evaluation.
    
    Returns:
    - metrics: Dictionary containing evaluation metrics.
    - y_pred: Predicted labels for the test data.
    """

    # Get model predictions (anomaly scores)
    anomaly_scores = model.predict(X_test)
    
    # Binarize predictions based on the threshold
    y_pred = (anomaly_scores > threshold).astype(int)
    
    # Precision, recall, F1-score
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    f1 = f1_score(y_test, y_pred, average=average)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC-AUC score
    if len(np.unique(y_test)) > 1:  # Ensure both classes are present
        auc = roc_auc_score(y_test, anomaly_scores)
    else:
        auc = np.nan  # AUC is undefined if only one class is present
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm
    }

    return metrics, y_pred


def plot_confusion_matrix(cm, model_name):
    """
    Plot the confusion matrix for a given model.
    
    Parameters:
    - cm: Confusion matrix.
    - model_name: Name of the model.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_roc_curve(y_test, anomaly_scores, model_name):
    """
    Plot ROC curve for a given model.
    
    Parameters:
    - y_test: True labels.
    - anomaly_scores: Predicted anomaly scores.
    - model_name: Name of the model.
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC: {roc_auc_score(y_test, anomaly_scores):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'ROC Curve: {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_multiple_models(models, X_test, y_test, threshold=0.5, average='binary', plot=False):
    """
    Evaluate multiple models and aggregate results.
    
    Parameters:
    - models: List of trained models.
    - X_test: Test data (time-series).
    - y_test: True labels (0 for normal, 1 for anomaly).
    - threshold: Decision threshold for anomaly scores.
    - average: Averaging method for multi-class or multi-label evaluation.
    - plot: Whether to plot confusion matrices and ROC curves.

    Returns:
    - results_df: DataFrame with evaluation metrics for each model.
    """

    results = []
    for model in models:
        metrics, y_pred = evaluate_model(model, X_test, y_test, threshold, average)
        model_name = model.__class__.__name__
        metrics['model'] = model_name
        results.append(metrics)
        
        if plot:
            plot_confusion_matrix(metrics['confusion_matrix'], model_name)
            plot_roc_curve(y_test, model.predict(X_test), model_name)
    
    results_df = pd.DataFrame(results)
    
    return results_df


def summarize_evaluation(results_df):
    """
    Summarize evaluation results by providing mean and standard deviation of metrics.
    
    Parameters:
    - results_df: DataFrame with evaluation metrics for multiple models.

    Returns:
    - summary_df: DataFrame summarizing the evaluation metrics.
    """
    
    summary_df = results_df.groupby('model').agg(['mean', 'std']).reset_index()
    
    return summary_df


def handle_imbalanced_data(X_train, y_train, method='oversample'):
    """
    Handle imbalanced dataset by either oversampling or undersampling.

    Parameters:
    - X_train: Training features.
    - y_train: Training labels.
    - method: Resampling method, either 'oversample' or 'undersample'.
    
    Returns:
    - resampled_X_train, resampled_y_train: Resampled training data and labels.
    """
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    if method == 'oversample':
        sampler = RandomOverSampler()
    else:
        sampler = RandomUnderSampler()
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled


def classification_report_extended(y_test, y_pred, average='binary'):
    """
    Generate a detailed classification report, including precision, recall, F1-score, and support.

    Parameters:
    - y_test: True labels.
    - y_pred: Predicted labels.
    - average: Averaging method for multi-class or multi-label evaluation.

    Returns:
    - report: Classification report as a string.
    """
    
    report = classification_report(y_test, y_pred, average=average)
    return report


def plot_metric_distribution(results_df, metric_name):
    """
    Plot the distribution of a specific metric across models.

    Parameters:
    - results_df: DataFrame containing evaluation metrics.
    - metric_name: The name of the metric to plot.
    """
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='model', y=metric_name, data=results_df)
    plt.title(f'Distribution of {metric_name} Across Models')
    plt.xticks(rotation=45)
    plt.show()


def custom_anomaly_thresholds(model, X_test, y_test, thresholds=[0.3, 0.5, 0.7]):
    """
    Evaluate the model with multiple custom thresholds for anomaly detection.

    Parameters:
    - model: Trained anomaly detection model.
    - X_test: Test data (time-series).
    - y_test: True labels.
    - thresholds: List of custom thresholds for anomaly scores.

    Returns:
    - threshold_results: DataFrame containing evaluation metrics for each threshold.
    """
    
    threshold_results = []
    
    for threshold in thresholds:
        metrics, _ = evaluate_model(model, X_test, y_test, threshold=threshold)
        metrics['threshold'] = threshold
        threshold_results.append(metrics)
    
    threshold_results_df = pd.DataFrame(threshold_results)
    
    return threshold_results_df


def generate_final_report(models, X_test, y_test):
    """
    Generate a comprehensive report by evaluating multiple models and plotting metrics.

    Parameters:
    - models: List of trained models.
    - X_test: Test data.
    - y_test: True labels.

    Returns:
    - final_report: Final evaluation report.
    """
    
    results_df = evaluate_multiple_models(models, X_test, y_test, plot=True)
    summary_df = summarize_evaluation(results_df)
    
    print("Evaluation Summary:")
    print(summary_df)
    
    # Plot metric distributions
    for metric in ['precision', 'recall', 'f1_score', 'roc_auc']:
        plot_metric_distribution(results_df, metric)
    
    return summary_df