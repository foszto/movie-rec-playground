# src/cli/train.py
import asyncio
import logging
import os
from pathlib import Path

import click
import torch.multiprocessing as mp
import yaml

from src.configs.logging_config import setup_logging
from src.configs.model_config import HybridConfig
from src.data.dataset import MovieLensDataset
from src.models.hybrid import HybridRecommender
from src.utils.data_io import load_preprocessed_data
from src.utils.visualization import plot_training_history


class EarlyStoppingHandler:
    """Enhanced early stopping with multiple metrics and smoothing."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, alpha: float = 0.9):
        self.patience = patience
        self.min_delta = min_delta
        self.alpha = alpha  # smoothing factor
        self.best_score = float("inf")
        self.counter = 0
        self.smoothed_metrics = {}

    def update_smoothed_metrics(self, metrics: dict) -> dict:
        """Update exponential moving average of metrics."""
        if not self.smoothed_metrics:
            self.smoothed_metrics = metrics.copy()
        else:
            for key, value in metrics.items():
                if key in self.smoothed_metrics:
                    self.smoothed_metrics[key] = (
                        self.alpha * self.smoothed_metrics[key]
                        + (1 - self.alpha) * value
                    )
        return self.smoothed_metrics

    def get_combined_score(self, metrics: dict) -> float:
        """Calculate combined score from multiple metrics."""
        smoothed = self.update_smoothed_metrics(metrics)
        return (
            smoothed.get("val_loss", 0) * 0.4
            + smoothed.get("val_mae", 0) * 0.3
            + smoothed.get("val_rmse", 0) * 0.3
        )

    def __call__(self, metrics: dict) -> bool:
        """Check if training should stop."""
        score = self.get_combined_score(metrics)

        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


# Set multiprocessing start method
mp.set_start_method("spawn", force=True)

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing preprocessed data",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Directory to save model and results",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to model configuration file",
)
def train(data_dir: str, output_dir: str, config_path: str):
    """Train the recommender model."""

    async def async_train():
        setup_logging()
        logger = logging.getLogger(__name__)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        logger.info("Loading preprocessed data...")
        (
            ratings_df,
            movies_df,
            tags_df,
            user_history_dict,
            movie_info_dict,
            user_tags_dict,
            movie_tags_dict,
        ) = load_preprocessed_data(data_dir)

        train_dataset, valid_dataset = MovieLensDataset.train_valid_split(
            ratings_df,
            valid_ratio=config.get("valid_ratio", 0.2),
            random_seed=config.get("random_seed", 42),
        )

        model_config = HybridConfig(
            model_type="hybrid_llm",
            n_users=train_dataset.n_users,
            n_items=train_dataset.n_items,
            **config.get("model_params", {}),
        )

        model = HybridRecommender(model_config)

        train_data = {
            "dataloader": train_dataset.get_dataloader(
                batch_size=config.get("batch_size", 32),
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            ),
            "movie_info_dict": movie_info_dict,
            "user_history_dict": user_history_dict,
            "user_tags_dict": user_tags_dict,
            "movie_tags_dict": movie_tags_dict,
        }

        valid_data = {
            "dataloader": valid_dataset.get_dataloader(
                batch_size=config.get("batch_size", 32),
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            ),
            "movie_info_dict": movie_info_dict,
            "user_history_dict": user_history_dict,
            "user_tags_dict": user_tags_dict,
            "movie_tags_dict": movie_tags_dict,
        }

        logger.info("Starting training...")
        history = []
        early_stopping = EarlyStoppingHandler(
            patience=config.get("early_stopping_patience", 5),
            min_delta=config.get("min_delta", 0.001),
            alpha=0.9,
        )

        for epoch in range(config.get("n_epochs", 10)):
            metrics = await model.fit(train_data, valid_data)
            history.append(metrics)

            # Log smoothed metrics
            smoothed = early_stopping.smoothed_metrics
            logger.info(
                f"Smoothed metrics - Val Loss: {smoothed.get('val_loss', 0):.4f}, "
                f"MAE: {smoothed.get('val_mae', 0):.4f}, "
                f"RMSE: {smoothed.get('val_rmse', 0):.4f}"
            )

            # Save model if improved
            combined_score = early_stopping.get_combined_score(metrics)
            if combined_score < early_stopping.best_score:
                logger.info(f"Model improved (score: {combined_score:.4f})")
                model.save(output_path / "best_model.pt")

            # Check early stopping
            if early_stopping(metrics):
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement in combined score for {early_stopping.patience} epochs)"
                )
                break

        model.save(output_path / "final_model.pt")
        plot_training_history(history, save_path=output_path / "training_history.png")

        logger.info("Training completed successfully.")

    asyncio.run(async_train())
