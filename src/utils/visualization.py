import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Union

def plot_training_history(history: Union[Dict[str, List[float]], List[Dict[str, float]]], save_path: str = None):
    """
    Plot training and validation metrics over time.
    
    Args:
        history: Either a dictionary of metric lists or a list of metric dictionaries
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Convert list of dicts to dict of lists if necessary
    if isinstance(history, list):
        metrics_dict = {}
        for epoch_metrics in history:
            for metric, value in epoch_metrics.items():
                if metric not in metrics_dict:
                    metrics_dict[metric] = []
                metrics_dict[metric].append(value)
        history = metrics_dict
    
    # Plot each metric
    for metric, values in history.items():
        plt.plot(values, label=metric)
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_rating_distribution(ratings: pd.Series, save_path: str = None):
    """Plot distribution of ratings."""
    plt.figure(figsize=(8, 6))
    sns.histplot(ratings, bins=10)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()