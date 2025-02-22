import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
from tqdm import tqdm

def calculate_metrics(predictions: np.ndarray, 
                     targets: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """Calculate various recommendation metrics with progress tracking."""
    metrics = {}
    
    # Progress bar for all metric calculations
    metric_steps = tqdm(
        total=7,  # Total number of metrics we'll calculate
        desc="Calculating metrics",
        position=0,
        leave=True
    )
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(targets, predictions)
    metric_steps.update(1)
    metric_steps.set_description("Calculating RMSE")
    
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metric_steps.update(1)
    metric_steps.set_description("Calculating MAE")
    
    metrics['mae'] = mean_absolute_error(targets, predictions)
    metric_steps.update(1)
    
    # Recommendation metrics
    metric_steps.set_description("Calculating Precision@K")
    metrics['precision'] = precision_at_k(predictions, targets, k=10)
    metric_steps.update(1)
    
    metric_steps.set_description("Calculating Recall@K")
    metrics['recall'] = recall_at_k(predictions, targets, k=10)
    metric_steps.update(1)
    
    metric_steps.set_description("Calculating NDCG@K")
    metrics['ndcg'] = ndcg_at_k(predictions, targets, k=10)
    metric_steps.update(1)
    
    metric_steps.set_description("Calculating MAP")
    metrics['map'] = mean_average_precision(predictions, targets)
    metric_steps.update(1)
    
    metric_steps.close()
    return metrics

def mean_average_precision(predictions: np.ndarray,
                         targets: np.ndarray,
                         sample_size: float = 0.01,  # 1% mintavételezés
                         random_seed: int = 42) -> float:
    """
    Calculate Mean Average Precision using sampling for large datasets.
    
    Args:
        predictions: Predicted ratings or scores
        targets: True ratings or relevance scores
        sample_size: Fraction of data to sample (0.01 = 1%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Approximated MAP score
    """
    # Sort by predictions
    sort_indices = np.argsort(predictions)[::-1]
    sorted_targets = targets[sort_indices]
    
    # Find relevant positions
    relevant_positions = np.where(sorted_targets >= 4.0)[0]
    if len(relevant_positions) == 0:
        return 0.0
    
    # If we have many relevant items, take a random sample
    if len(relevant_positions) > 1000:  # Only sample for large datasets
        np.random.seed(random_seed)
        sample_size = int(len(relevant_positions) * sample_size)
        sample_size = max(100, min(sample_size, 10000))  # Min 100, max 10000 samples
        
        # Stratified sampling: take some from start, middle, and end
        n_each = sample_size // 3
        start_idx = relevant_positions[:n_each]
        mid_idx = relevant_positions[len(relevant_positions)//2 - n_each//2:len(relevant_positions)//2 + n_each//2]
        end_idx = relevant_positions[-n_each:]
        sampled_positions = np.concatenate([start_idx, mid_idx, end_idx])
        
        # Add some random positions
        remaining = sample_size - len(sampled_positions)
        if remaining > 0:
            mask = ~np.isin(relevant_positions, sampled_positions)
            pool = relevant_positions[mask]
            if len(pool) > 0:
                random_idx = np.random.choice(pool, size=remaining, replace=False)
                sampled_positions = np.concatenate([sampled_positions, random_idx])
        
        relevant_positions = np.sort(sampled_positions)
        
    # Calculate precision for sampled positions
    precisions = []
    progress = tqdm(
        enumerate(relevant_positions, 1),
        total=len(relevant_positions),
        desc=f"Calculating MAP (sampling {len(relevant_positions)} items)",
        position=1,
        leave=False
    )
    
    for i, pos in progress:
        precision = np.sum(sorted_targets[:pos + 1] >= 4.0) / (pos + 1)
        precisions.append(precision)
        if i % 10 == 0:  # Update less frequently
            progress.set_description(
                f"MAP (sampled, current avg: {np.mean(precisions):.3f})"
            )
    
    return np.mean(precisions)

def ndcg_at_k(predictions: np.ndarray,
              targets: np.ndarray,
              k: int = 10) -> float:
    """Calculate NDCG@K with progress tracking for large arrays."""
    if len(predictions) > 10000:
        progress = tqdm(
            total=4,
            desc="Computing NDCG",
            position=1,
            leave=False
        )
        
        # Sort predictions
        progress.set_description("Sorting predictions")
        pred_order = np.argsort(predictions)[::-1]
        progress.update(1)
        
        progress.set_description("Sorting targets")
        true_order = np.argsort(targets)[::-1]
        progress.update(1)
        
        # Get relevance scores
        progress.set_description("Computing relevance scores")
        pred_relevance = targets[pred_order]
        true_relevance = targets[true_order]
        progress.update(1)
        
        # Calculate DCG
        progress.set_description("Computing DCG values")
        pred_dcg = dcg_at_k(pred_relevance, k)
        true_dcg = dcg_at_k(true_relevance, k)
        progress.update(1)
        progress.close()
    else:
        # For smaller arrays, calculate without progress tracking
        pred_order = np.argsort(predictions)[::-1]
        true_order = np.argsort(targets)[::-1]
        pred_relevance = targets[pred_order]
        true_relevance = targets[true_order]
        pred_dcg = dcg_at_k(pred_relevance, k)
        true_dcg = dcg_at_k(true_relevance, k)
    
    return pred_dcg / true_dcg if true_dcg > 0 else 0.0

def diversity_at_k(predictions: np.ndarray,
                  item_features: np.ndarray,
                  k: int = 10) -> float:
    """Calculate diversity with progress tracking for large feature matrices."""
    top_k_indices = np.argsort(predictions)[::-1][:k]
    top_k_features = item_features[top_k_indices]
    
    # Show progress only for large feature matrices
    if k > 100:
        progress = tqdm(
            total=2,
            desc="Computing diversity",
            position=1,
            leave=False
        )
        
        # Calculate similarity matrix
        progress.set_description("Computing similarity matrix")
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(top_k_features)
        progress.update(1)
        
        # Calculate diversity
        progress.set_description("Computing final diversity score")
        n = len(sim_matrix)
        diversity = 0.0
        if n > 1:
            diversity = (1 - sim_matrix.sum() + n) / (n * (n - 1))
        progress.update(1)
        progress.close()
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(top_k_features)
        n = len(sim_matrix)
        diversity = 0.0
        if n > 1:
            diversity = (1 - sim_matrix.sum() + n) / (n * (n - 1))
    
    return diversity

# Egyszerűbb metrikák változatlanul maradnak
def precision_at_k(predictions: np.ndarray,
                  targets: np.ndarray,
                  k: int = 10) -> float:
    top_k_indices = np.argsort(predictions)[::-1][:k]
    relevant_in_k = np.sum(targets[top_k_indices] >= 4.0)
    return relevant_in_k / k if k > 0 else 0.0

def recall_at_k(predictions: np.ndarray,
                targets: np.ndarray,
                k: int = 10) -> float:
    top_k_indices = np.argsort(predictions)[::-1][:k]
    total_relevant = np.sum(targets >= 4.0)
    relevant_in_k = np.sum(targets[top_k_indices] >= 4.0)
    return relevant_in_k / total_relevant if total_relevant > 0 else 0.0

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    relevance = np.asarray(relevance)[:k]
    n_rel = len(relevance)
    if n_rel == 0:
        return 0.0
    discounts = np.log2(np.arange(2, n_rel + 2))
    return np.sum(relevance / discounts)

def hit_rate_at_k(predictions: np.ndarray,
                  targets: np.ndarray,
                  k: int = 10) -> float:
    top_k_indices = np.argsort(predictions)[::-1][:k]
    hits = np.sum(targets[top_k_indices] >= 4.0) > 0
    return float(hits)