import torch
import torch.nn as nn


class DiversityAwareLoss(nn.Module):
    """Loss function that encourages diverse recommendations."""

    def __init__(self, diversity_lambda: float = 0.25, rating_lambda: float = 0.2):
        super().__init__()
        self.diversity_lambda = diversity_lambda
        self.rating_lambda = rating_lambda
        self.base_criterion = nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss with diversity penalty.

        Args:
            pred: Predicted ratings
            target: True ratings
            item_ids: Movie IDs for the batch

        Returns:
            Combined loss value
        """
        # Base prediction loss
        base_loss = self.base_criterion(pred, target)

        # Calculate per-item statistics
        unique_items, inverse_indices, counts = torch.unique(
            item_ids, return_inverse=True, return_counts=True
        )

        # Get per-item average predictions
        item_predictions = (
            torch.zeros(len(unique_items), device=pred.device).scatter_add_(
                0, inverse_indices, pred
            )
            / counts.float()
        )

        # Diversity loss: penalize predictions that are too similar
        pred_std = torch.std(item_predictions)
        diversity_loss = torch.exp(-pred_std)  # Exponential penalty for low diversity

        # Rating distribution loss: encourage using full rating range
        rating_range = torch.max(pred) - torch.min(pred)
        rating_loss = torch.exp(-rating_range)  # Penalize narrow rating ranges

        # Combine losses
        total_loss = (
            base_loss
            + self.diversity_lambda * diversity_loss
            + self.rating_lambda * rating_loss
        )

        return total_loss
