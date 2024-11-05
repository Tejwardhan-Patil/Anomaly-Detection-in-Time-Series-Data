import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
from utils.visualization import plot_time_series_with_anomalies

# Load evaluation metrics from JSON files
def load_metrics(metrics_dir):
    metrics = {}
    for file_name in os.listdir(metrics_dir):
        if file_name.endswith(".json"):
            model_name = file_name.split(".")[0]
            with open(os.path.join(metrics_dir, file_name)) as file:
                metrics[model_name] = json.load(file)
    return metrics

# Generate HTML report using Jinja2 templates
def generate_html_report(metrics, output_dir, report_template):
    with open(report_template, 'r') as file:
        template = Template(file.read())

    rendered_html = template.render(metrics=metrics)
    
    report_path = os.path.join(output_dir, 'evaluation_report.html')
    with open(report_path, 'w') as report_file:
        report_file.write(rendered_html)
    print(f"HTML report generated at: {report_path}")

# Save summary of metrics to CSV
def save_summary_csv(metrics, output_dir):
    summary_data = []
    for model_name, model_metrics in metrics.items():
        summary_data.append({
            "Model": model_name,
            "Precision": model_metrics["precision"],
            "Recall": model_metrics["recall"],
            "F1-Score": model_metrics["f1_score"],
            "AUC": model_metrics.get("auc", None)
        })

    df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'model_metrics_summary.csv')
    df.to_csv(summary_csv_path, index=False)
    print(f"Metrics summary CSV generated at: {summary_csv_path}")

# Generate plots for anomaly detection results
def generate_plots(metrics, data_dir, output_dir):
    for model_name, model_metrics in metrics.items():
        anomalies = model_metrics.get("anomalies", [])
        data_file = os.path.join(data_dir, f"{model_name}_data.csv")

        if os.path.exists(data_file):
            df = pd.read_csv(data_file, parse_dates=["timestamp"])
            fig, ax = plt.subplots()
            plot_time_series_with_anomalies(df, anomalies, ax)
            plot_path = os.path.join(output_dir, f"{model_name}_anomaly_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Anomaly plot for {model_name} saved at: {plot_path}")
        else:
            print(f"Data file for {model_name} not found. Skipping plot generation.")

# Main function to execute the report generation
def main(metrics_dir, data_dir, output_dir, report_template):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load evaluation metrics
    metrics = load_metrics(metrics_dir)

    # Generate HTML report
    generate_html_report(metrics, output_dir, report_template)

    # Save summary to CSV
    save_summary_csv(metrics, output_dir)

    # Generate anomaly detection plots
    generate_plots(metrics, data_dir, output_dir)

# Usage
if __name__ == "__main__":
    metrics_dir = "/metrics"
    data_dir = "/data"
    output_dir = "/output/reports"
    report_template = "/templates/report_template.html"
    
    main(metrics_dir, data_dir, output_dir, report_template)