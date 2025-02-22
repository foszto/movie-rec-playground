import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple

def calculate_metrics(predictions: np.ndarray, 
                     targets: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate various recommendation metrics.
    
    Args:
        predictions: Predicted ratings or scores
        targets: True ratings or relevance scores
        threshold: Threshold for considering an item relevant
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        'mse': mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions)
    }
    
    # Add additional recommendation-specific metrics
    metrics['precision'] = precision_at_k(predictions, targets, k=10)
    metrics['recall'] = recall_at_k(predictions, targets, k=10)
    metrics['ndcg'] = ndcg_at_k(predictions, targets, k=10)
    metrics['map'] = mean_average_precision(predictions, targets)
    
    return metrics

def precision_at_k(predictions: np.ndarray,
                  targets: np.ndarray,
                  k: int = 10) -> float:
    """
    Calculate Precision@K metric.
    
    Args:
        predictions: Predicted ratings or scores
        targets: True ratings or relevance scores
        k: Number of items to consider
        
    Returns:
        Precision@K score
    """
    # Sort predictions in descending order and get top k indices
    top_k_indices = np.argsort(predictions)[::-1][:k]
    
    # Count relevant items in top k
    relevant_in_k = np.sum(targets[top_k_indices] >= 4.0)  # Consider ratings >= 4 as relevant
    
    return relevant_in_k / k if k > 0 else 0.0

def recall_at_k(predictions: np.ndarray,
                targets: np.ndarray,
                k: int = 10) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        predictions: Predicted ratings or scores
        targets: True ratings or relevance scores
        k: Number of items to consider
        
    Returns:
        Recall@K score
    """
    # Get indices of top k predicted items
    top_k_indices = np.argsort(predictions)[::-1][:k]
    
    # Count relevant items
    total_relevant = np.sum(targets >= 4.0)  # Consider ratings >= 4 as relevant
    relevant_in_k = np.sum(targets[top_k_indices] >= 4.0)
    
    return relevant_in_k / total_relevant if total_relevant > 0 else 0.0

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.
    
    Args:
        relevance: Array of relevance scores
        k: Number of items to consider
        
    Returns:
        DCG@K score
    """
    relevance = np.asarray(relevance)[:k]
    n_rel = len(relevance)
    if n_rel == 0:
        return 0.0
        
    discounts = np.log2(np.arange(2, n_rel + 2))
    return np.sum(relevance / discounts)

def ndcg_at_k(predictions: np.ndarray,
              targets: np.ndarray,
              k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) metric.
    
    Args:
        predictions: Predicted ratings or scores
        targets: True ratings or relevance scores
        k: Number of items to consider
        
    Returns:
        NDCG@K score
    """
    # Sort predictions and get the order
    pred_order = np.argsort(predictions)[::-1]
    true_order = np.argsort(targets)[::-1]
    
    # Get relevance scores in both orders
    pred_relevance = targets[pred_order]
    true_relevance = targets[true_order]
    
    # Calculate DCG for both
    pred_dcg = dcg_at_k(pred_relevance, k)
    true_dcg = dcg_at_k(true_relevance, k)
    
    return pred_dcg / true_dcg if true_dcg > 0 else 0.0

def mean_average_precision(predictions: np.ndarray,
                         targets: np.ndarray) -> float:
    """
    Calculate Mean Average Precision.
    
    Args:
        predictions: Predicted ratings or scores
        targets: True ratings or relevance scores
        
    Returns:
        MAP score
    """
    # Sort by predictions
    sort_indices = np.argsort(predictions)[::-1]
    sorted_targets = targets[sort_indices]
    
    # Calculate precision at each position for relevant items
    relevant_positions = np.where(sorted_targets >= 4.0)[0]
    if len(relevant_positions) == 0:
        return 0.0
    
    precisions = []
    for i, pos in enumerate(relevant_positions, 1):
        precisions.append(np.sum(sorted_targets[:pos + 1] >= 4.0) / (pos + 1))
    
    return np.mean(precisions)

def hit_rate_at_k(predictions: np.ndarray,
                  targets: np.ndarray,
                  k: int = 10) -> float:
    """
    Calculate Hit Rate@K metric.
    
    Args:
        predictions: Predicted ratings or scores
        targets: True ratings or relevance scores
        k: Number of items to consider
        
    Returns:
        Hit Rate@K score
    """
    top_k_indices = np.argsort(predictions)[::-1][:k]
    hits = np.sum(targets[top_k_indices] >= 4.0) > 0
    return float(hits)

def diversity_at_k(predictions: np.ndarray,
                  item_features: np.ndarray,
                  k: int = 10) -> float:
    """
    Calculate diversity of top-K recommendations.
    
    Args:
        predictions: Predicted ratings or scores
        item_features: Feature vectors for items
        k: Number of items to consider
        
    Returns:
        Diversity score
    """
    top_k_indices = np.argsort(predictions)[::-1][:k]
    top_k_features = item_features[top_k_indices]
    
    # Calculate pairwise distances
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(top_k_features)
    
    # Average dissimilarity (1 - similarity)
    n = len(sim_matrix)
    diversity = 0.0
    if n > 1:
        diversity = (1 - sim_matrix.sum() + n) / (n * (n - 1))  # Excluding self-similarity
    
    return diversity