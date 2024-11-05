import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns

# Configure logging
logging.basicConfig(filename='random_forest_model.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load preprocessed time-series data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        logging.info("Data loaded successfully from {}".format(filepath))
        return data
    except Exception as e:
        logging.error("Error loading data: {}".format(e))
        raise

# Split data into features and labels
def split_features_labels(data, label_column="anomaly_label"):
    X = data.drop(columns=[label_column])
    y = data[label_column]
    logging.info("Data split into features and labels")
    return X, y

# Standardize the dataset
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Data standardized")
    return X_train_scaled, X_test_scaled

# Train Random Forest model
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    logging.info("Random Forest model trained with n_estimators={}, max_depth={}".format(n_estimators, max_depth))
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logging.info("Model evaluation completed: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(
        precision, recall, f1, auc))

    return precision, recall, f1, auc, cm

# Plot confusion matrix
def plot_confusion_matrix(cm, classes=['Normal', 'Anomaly'], title='Confusion Matrix'):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Perform k-fold cross-validation
def cross_validate_model(model, X, y, k=5):
    kfold = StratifiedKFold(n_splits=k)
    results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
    logging.info("Cross-validation F1 scores: {}".format(results))
    return results

# Save model
def save_model(model, output_dir="models", filename="random_forest_model.pkl"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    pd.to_pickle(model, filepath)
    logging.info("Model saved at {}".format(filepath))

# Feature importance plotting
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Main function to train, evaluate, and save the model
def main(filepath):
    # Load the data
    data = load_data(filepath)
    logging.info("Data loaded, starting preprocessing...")

    # Split data into features and labels
    X, y = split_features_labels(data)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets")
    
    # Standardize the data
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # Train the model
    model = train_random_forest(X_train_scaled, y_train)

    # Evaluate the model
    precision, recall, f1, auc, cm = evaluate_model(model, X_test_scaled, y_test)
    logging.info(f"Model Performance: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}, AUC={auc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(cm)

    # Perform cross-validation
    cross_val_f1_scores = cross_validate_model(model, X_train_scaled, y_train, k=5)
    logging.info("Cross-validation F1-Score: {:.4f} ± {:.4f}".format(np.mean(cross_val_f1_scores), np.std(cross_val_f1_scores)))

    # Feature Importance
    feature_names = X.columns
    plot_feature_importance(model, feature_names)

    # Save the model
    save_model(model)

# Run the script
if __name__ == "__main__":
    filepath = "/processed/data.csv"
    main(filepath)