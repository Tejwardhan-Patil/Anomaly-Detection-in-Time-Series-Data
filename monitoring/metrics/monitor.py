import os
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Logger setup
log_dir = "monitoring/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'performance_metrics.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class PerformanceMonitor:
    def __init__(self, alert_email=None, precision_threshold=0.9, recall_threshold=0.9, f1_threshold=0.9):
        self.metrics_history = []
        self.alert_email = alert_email
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        self.f1_threshold = f1_threshold

    def send_alert(self, subject, message):
        if self.alert_email:
            try:
                email = "monitoring@website.com"  # Define email for alert
                msg = MIMEMultipart()
                msg['From'] = email
                msg['To'] = self.alert_email
                msg['Subject'] = subject
                msg.attach(MIMEText(message, 'plain'))

                # Connect and send email
                server = smtplib.SMTP('smtp.website.com', 587)
                server.starttls()
                server.login(email, "password")  # Email login
                text = msg.as_string()
                server.sendmail(email, self.alert_email, text)
                server.quit()
                print("Alert sent to", self.alert_email)
            except Exception as e:
                print("Failed to send alert:", e)

    def check_alerts(self, precision, recall, f1):
        alerts_triggered = []
        if precision < self.precision_threshold:
            alerts_triggered.append(f"Low Precision: {precision:.4f}")
        if recall < self.recall_threshold:
            alerts_triggered.append(f"Low Recall: {recall:.4f}")
        if f1 < self.f1_threshold:
            alerts_triggered.append(f"Low F1-Score: {f1:.4f}")

        if alerts_triggered:
            alert_message = "\n".join(alerts_triggered)
            self.send_alert("Model Performance Alert", alert_message)

    def log_metrics(self, y_true, y_pred):
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Log metrics
        metrics = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }
        logging.info(json.dumps(metrics))
        self.metrics_history.append(metrics)

        # Check and trigger alerts if any thresholds are breached
        self.check_alerts(precision, recall, f1)

        # Print metrics for immediate inspection
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, False Positives: {fp}, False Negatives: {fn}")
        
    def plot_metrics_over_time(self, show_fp_fn=True):
        # Plot metrics history
        timestamps = [entry['timestamp'] for entry in self.metrics_history]
        precisions = [entry['precision'] for entry in self.metrics_history]
        recalls = [entry['recall'] for entry in self.metrics_history]
        f1_scores = [entry['f1_score'] for entry in self.metrics_history]
        
        plt.figure(figsize=(12, 8))

        # Precision, Recall, F1-Score plot
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, precisions, label='Precision', marker='o', color='green')
        plt.plot(timestamps, recalls, label='Recall', marker='o', color='blue')
        plt.plot(timestamps, f1_scores, label='F1-Score', marker='o', color='orange')
        plt.xticks(rotation=45, ha='right')
        plt.title('Model Performance Metrics Over Time')
        plt.legend()
        plt.grid(True)

        # False Positives and False Negatives plot
        if show_fp_fn:
            fps = [entry['false_positives'] for entry in self.metrics_history]
            fns = [entry['false_negatives'] for entry in self.metrics_history]
            plt.subplot(3, 1, 2)
            plt.plot(timestamps, fps, label='False Positives', color='red', marker='x')
            plt.plot(timestamps, fns, label='False Negatives', color='purple', marker='x')
            plt.xticks(rotation=45, ha='right')
            plt.title('False Positives and False Negatives')
            plt.legend()
            plt.grid(True)

        # True Positives and True Negatives plot
        tps = [entry['true_positives'] for entry in self.metrics_history]
        tns = [entry['true_negatives'] for entry in self.metrics_history]
        plt.subplot(3, 1, 3)
        plt.plot(timestamps, tps, label='True Positives', color='cyan', marker='s')
        plt.plot(timestamps, tns, label='True Negatives', color='magenta', marker='s')
        plt.xticks(rotation=45, ha='right')
        plt.title('True Positives and True Negatives')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_metrics_history(self, file_path="monitoring/metrics_history.json"):
        with open(file_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        print(f"Metrics history saved to {file_path}")

    def load_metrics_history(self, file_path="monitoring/metrics_history.json"):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.metrics_history = json.load(f)
            print(f"Metrics history loaded from {file_path}")
        else:
            print(f"No metrics history file found at {file_path}")

if __name__ == "__main__":
    monitor = PerformanceMonitor(alert_email="admin@website.com", precision_threshold=0.85, recall_threshold=0.85, f1_threshold=0.85)

    # Usage:
    # y_true and y_pred should be passed from the model's predictions during inference
    # y_true = [actual labels]
    # y_pred = [model predictions]
    
    # Log the metrics
    # monitor.log_metrics(y_true, y_pred)

    # Plot metrics over time
    # monitor.plot_metrics_over_time()

    # Save metrics history to a file
    # monitor.save_metrics_history()

    # Load metrics history from a file
    # monitor.load_metrics_history()