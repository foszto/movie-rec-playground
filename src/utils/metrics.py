import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

def normalize_predictions(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Normalize predictions to match the target value range.
    
    Args:
        predictions: Raw model predictions
        targets: True target values
        
    Returns:
        Normalized predictions matching target scale
    """
    pred_min, pred_max = predictions.min(), predictions.max()
    target_min, target_max = targets.min(), targets.max()
    
    normalized = (predictions - pred_min) * (target_max - target_min) / (pred_max - pred_min) + target_min
    
    # Clip values to ensure they stay within target range
    return np.clip(normalized, target_min, target_max)

def calculate_metrics(predictions: np.ndarray, 
                     targets: np.ndarray,
                     threshold: float = 3.5,
                     normalize: bool = True) -> Dict[str, float]:
    """
    Calculate metrics with optional prediction normalization.
    """
    # Normalize predictions if requested
    if normalize:
        predictions = normalize_predictions(predictions, targets)
    
    metrics = {}
    
    # Progress bar for all metric calculations
    metric_steps = tqdm(
        total=7,
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
    
    # Distribution statistics
    metrics['pred_mean'] = np.mean(predictions)
    metrics['pred_std'] = np.std(predictions)
    metrics['target_mean'] = np.mean(targets)
    metrics['target_std'] = np.std(targets)
    
    # Recommendation metrics
    metric_steps.set_description("Calculating Precision@K")
    metrics['precision'] = precision_at_k(predictions, targets, k=10, threshold=threshold)
    metric_steps.update(1)
    
    metric_steps.set_description("Calculating Recall@K")
    metrics['recall'] = recall_at_k(predictions, targets, k=10, threshold=threshold)
    metric_steps.update(1)
    
    metric_steps.set_description("Calculating NDCG@K")
    metrics['ndcg'] = ndcg_at_k(predictions, targets, k=10)
    metric_steps.update(1)
    
    metric_steps.set_description("Calculating MAP")
    metrics['map'] = mean_average_precision(predictions, targets, threshold=threshold)
    metric_steps.update(1)
    
    metric_steps.close()
    return metrics

def mean_average_precision(predictions: np.ndarray,
                         targets: np.ndarray,
                         threshold: float = 4.0,
                         sample_size: float = 0.01) -> float:
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
    relevant_positions = np.where(sorted_targets >= threshold)[0]
    if len(relevant_positions) == 0:
        return 0.0
    
    # If we have many relevant items, take a random sample
    if len(relevant_positions) > 1000:
        sample_size = int(len(relevant_positions) * sample_size)
        sample_size = max(100, min(sample_size, 1000))
        
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
        precision = np.sum(sorted_targets[:pos + 1] >= threshold) / (pos + 1)
        precisions.append(precision)
        if i % 10 == 0: # Update less frequently
            progress.set_description(
                f"MAP (sampled, current avg: {np.mean(precisions):.3f})"
            )
    
    return np.mean(precisions)

# Helper function to analyze prediction distribution
def analyze_predictions(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Analyze prediction distribution compared to targets."""
    return {
        'pred_min': predictions.min(),
        'pred_max': predictions.max(),
        'pred_mean': predictions.mean(),
        'pred_std': predictions.std(),
        'target_min': targets.min(),
        'target_max': targets.max(),
        'target_mean': targets.mean(),
        'target_std': targets.std(),
        'correlation': np.corrcoef(predictions, targets)[0, 1]
    }

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

# Simple metrics for debugging

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



def debug_metrics(predictions: np.ndarray, 
                   targets: np.ndarray,
                   k: int = 10,
                   threshold: float = 4.0) -> Dict[str, Any]:
    """Debug metric calculations with detailed intermediate results."""
    debug_info = {}
    
    # Sort by predictions
    sort_indices = np.argsort(predictions)[::-1]
    
    # Get top k items
    top_k_indices = sort_indices[:k]
    
    # Get values for these items
    debug_info['top_k_predictions'] = predictions[top_k_indices]
    debug_info['top_k_targets'] = targets[top_k_indices]
    
    # Count relevant items
    total_relevant = np.sum(targets >= threshold)
    relevant_in_k = np.sum(targets[top_k_indices] >= threshold)
    
    debug_info['total_relevant'] = total_relevant
    debug_info['relevant_in_k'] = relevant_in_k
    debug_info['threshold'] = threshold
    
    # Calculate metrics
    debug_info['precision'] = relevant_in_k / k if k > 0 else 0.0
    debug_info['recall'] = relevant_in_k / total_relevant if total_relevant > 0 else 0.0
    
    # Distribution info
    debug_info['n_predictions_above_threshold'] = np.sum(predictions >= threshold)
    debug_info['n_targets_above_threshold'] = total_relevant
    
    return debug_info

def print_metric_debug(debug_info: Dict[str, Any]) -> None:
    """Print debug information in a readable format."""
    print("\nMetric Debug Information:")
    print("-" * 50)
    print(f"Threshold: {debug_info['threshold']}")
    print("\nTop K Predictions vs Targets:")
    for i, (pred, target) in enumerate(zip(debug_info['top_k_predictions'], 
                                         debug_info['top_k_targets'])):
        print(f"Item {i+1}: Pred={pred:.2f}, Target={target:.2f}")
    
    print("\nRelevance Counts:")
    print(f"Total relevant items: {debug_info['total_relevant']}")
    print(f"Relevant items in top K: {debug_info['relevant_in_k']}")
    print(f"Predictions above threshold: {debug_info['n_predictions_above_threshold']}")
    print(f"Targets above threshold: {debug_info['n_targets_above_threshold']}")
    
    print("\nMetrics:")
    print(f"Precision@K: {debug_info['precision']:.4f}")
    print(f"Recall@K: {debug_info['recall']:.4f}")

def normalize_ratings(ratings: np.ndarray, min_rating: float = 1.0, max_rating: float = 5.0) -> np.ndarray:
    """
    Normalize ratings to a fixed scale.
    
    Args:
        ratings: Raw rating values
        min_rating: Minimum possible rating value
        max_rating: Maximum possible rating value
        
    Returns:
        Normalized ratings in [0, 1] range
    """
    return (ratings - min_rating) / (max_rating - min_rating)

def are_ratings_equal(pred: float, target: float, tolerance: float = 0.25) -> bool:
    """
    Check if prediction matches target within tolerance.
    A prediction is considered correct if it rounds to the same value as the target.
    """
    rounded_pred = round_ratings(pred)
    rounded_target = round_ratings(target)
    return rounded_pred == rounded_target

def precision_at_k(predictions: np.ndarray,
                           targets: np.ndarray,
                           k: int = 10,
                           threshold: float = 4.0) -> float:
    """
    Precision@K using custom rating rounding rules.
    """
    top_k_indices = np.argsort(predictions)[::-1][:k]
    good_predictions = np.array([
        are_ratings_equal(pred, target)
        for pred, target in zip(predictions[top_k_indices], targets[top_k_indices])
    ])
    return np.sum(good_predictions) / k if k > 0 else 0.0

def recall_at_k(predictions: np.ndarray,
                 targets: np.ndarray,
                 k: int = 10,
                 threshold: float = 4.0) -> float:
    """
    Calculate Recall@K for recommender system evaluation.
    
    Args:
        predictions: Predicted ratings/scores for items
        targets: True ratings for items
        k: Number of top items to consider
        threshold: Rating threshold above which an item is considered relevant
        
    Returns:
        Recall@K score between 0 and 1
        
    Example:
        If k=10, threshold=4.0:
        - Total relevant items (target >= 4.0): 100
        - In top 10 predictions, 5 items have target rating >= 4.0
        - Recall@10 = 5/100 = 0.05
    """
    # Find total number of relevant items
    total_relevant = np.sum(targets >= threshold)
    
    if total_relevant == 0:
        return 0.0
        
    # Get indices of top k predictions
    top_k_indices = np.argsort(predictions)[::-1][:k]
    
    # Count how many relevant items we found in top k
    relevant_in_top_k = np.sum(targets[top_k_indices] >= threshold)
    
    # Calculate recall
    recall = relevant_in_top_k / total_relevant
    
    return recall

def analyze_rating_distribution(predictions: np.ndarray,
                              targets: np.ndarray,
                              rating_scale: Tuple[float, float] = (1.0, 5.0)) -> Dict[str, Any]:
    """
    Analyze rating distributions after normalization.
    """
    min_rating, max_rating = rating_scale
    norm_predictions = normalize_ratings(predictions, min_rating, max_rating)
    norm_targets = normalize_ratings(targets, min_rating, max_rating)
    
    return {
        'norm_pred_mean': norm_predictions.mean(),
        'norm_pred_std': norm_predictions.std(),
        'norm_target_mean': norm_targets.mean(),
        'norm_target_std': norm_targets.std(),
        'pred_unique_values': len(np.unique(norm_predictions)),
        'target_unique_values': len(np.unique(norm_targets)),
        'correlation': np.corrcoef(norm_predictions, norm_targets)[0, 1]
    }

def calc_thresholded_stats(predictions: np.ndarray,
                          targets: np.ndarray,
                          threshold: float = 4.0,
                          rating_scale: Tuple[float, float] = (1.0, 5.0)) -> Dict[str, float]:
    """
    Calculate statistics about ratings above/below threshold after normalization.
    """
    min_rating, max_rating = rating_scale
    norm_predictions = normalize_ratings(predictions, min_rating, max_rating)
    norm_targets = normalize_ratings(targets, min_rating, max_rating)
    norm_threshold = normalize_ratings(np.array([threshold]), min_rating, max_rating)[0]
    
    pred_above = norm_predictions >= norm_threshold
    target_above = norm_targets >= norm_threshold
    
    return {
        'pred_above_threshold': np.mean(pred_above),
        'target_above_threshold': np.mean(target_above),
        'agreement_rate': np.mean(pred_above == target_above),
        'false_positives': np.mean(pred_above & ~target_above),
        'false_negatives': np.mean(~pred_above & target_above)
    }

def round_ratings(values):
    """
    Round ratings to nearest 0.5 increment.
    Handles both scalar values and numpy arrays.
    
    Args:
        values: Single value or numpy array of values to round
        
    Returns:
        Rounded value(s) to nearest 0.5
    """
    if isinstance(values, np.ndarray):
        return np.round(values * 2) / 2
    else:
        return round(values * 2) / 2