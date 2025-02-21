import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot training and validation metrics over time."""
    plt.figure(figsize=(10, 6))
    
    for metric, values in history.items():
        plt.plot(values, label=metric)
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
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
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()