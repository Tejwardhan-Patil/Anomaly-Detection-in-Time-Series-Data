import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

def plot_time_series(data, title="Time-Series Data", xlabel="Time", ylabel="Value"):
    """
    Plots the time-series data.

    Parameters:
    - data: Pandas DataFrame or Series containing the time-series data to plot.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_anomalies(data, anomalies, title="Detected Anomalies", xlabel="Time", ylabel="Value"):
    """
    Plots time-series data with detected anomalies.

    Parameters:
    - data: Pandas DataFrame or Series containing the time-series data.
    - anomalies: Boolean or binary series indicating the anomaly points.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, label="Time-Series Data")
    plt.scatter(data.index[anomalies], data[anomalies], color='red', label="Anomalies", zorder=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_interactive_time_series(data, anomalies=None, title="Interactive Time-Series Plot"):
    """
    Creates an interactive time-series plot using Plotly.

    Parameters:
    - data: Pandas DataFrame or Series containing the time-series data.
    - anomalies: Boolean or binary series indicating the anomaly points (Optional).
    - title: Title of the interactive plot.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name='Time-Series Data'))
    
    if anomalies is not None:
        fig.add_trace(go.Scatter(x=data.index[anomalies], y=data[anomalies], mode='markers', name='Anomalies', marker=dict(color='red', size=8)))

    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value", template="plotly_white")
    fig.show()

def plot_correlation_matrix(data, title="Correlation Matrix"):
    """
    Plots a heatmap of the correlation matrix for the given data.

    Parameters:
    - data: Pandas DataFrame containing time-series data.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title(title)
    plt.show()