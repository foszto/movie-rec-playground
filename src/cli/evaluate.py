import asyncio
import logging
from pathlib import Path

import click
import pandas as pd
import torch
from torch.serialization import add_safe_globals, safe_globals
from tqdm import tqdm

from src.configs.logging_config import setup_logging
from src.configs.model_config import HybridConfig
from src.data.dataset import MovieLensDataset
from src.models.hybrid import HybridRecommender
from src.utils.data_io import load_preprocessed_data
from src.utils.metrics import calculate_metrics


async def evaluate_model_with_progress(model, test_data, device):
    """Evaluate model with progress bar, keeping tensors on GPU for efficiency."""
    predictions_list = []
    targets_list = []

    with torch.no_grad():
        total_batches = len(test_data["dataloader"])
        progress_bar = tqdm(
            test_data["dataloader"],
            total=total_batches,
            desc="Evaluating batches",
            position=0,
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            ratings = batch["rating"].to(device)

            predictions = await model.predict(
                user_ids,
                item_ids,
                test_data["movie_info_dict"],
                test_data["user_history_dict"],
            )

            predictions_list.append(predictions)
            targets_list.append(ratings)

            avg_rating = torch.mean(predictions).item()
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{total_batches} [Avg Rating: {avg_rating:.2f}]"
            )

        all_predictions = torch.cat(predictions_list)
        all_targets = torch.cat(targets_list)

        return all_predictions.cpu().numpy(), all_targets.cpu().numpy()


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing test data",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Directory to save evaluation results",
)
@click.option("--batch-size", type=int, default=128, help="Batch size for evaluation")
def evaluate(data_dir: str, model_path: str, output_dir: str, batch_size: int):
    """Evaluate the trained model."""

    setup_logging()
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load data with progress indication
        logger.info("Loading test data...")
        data_loading = tqdm(total=1, desc="Loading dataset", position=0, leave=True)
        (
            ratings_df,
            movies_df,
            tags_df,
            user_history_dict,
            movie_info_dict,
            user_tags_dict,
            movie_tags_dict,
        ) = load_preprocessed_data(data_dir)
        data_loading.update(1)
        data_loading.close()

        # Create test dataset
        logger.info("Creating test dataset...")
        test_dataset = MovieLensDataset(ratings_df)
        test_data = {
            "dataloader": test_dataset.get_dataloader(
                batch_size=batch_size, shuffle=False
            ),
            "movie_info_dict": movie_info_dict,
            "user_history_dict": user_history_dict,
            "user_tags_dict": user_tags_dict,
            "movie_tags_dict": movie_tags_dict,
        }

        # Load model with progress
        logger.info("Loading model...")
        model_loading = tqdm(total=1, desc="Loading model", position=0, leave=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_file = list(Path(model_path).glob("*.pt"))[0]

        add_safe_globals([HybridConfig])
        with safe_globals([HybridConfig]):
            checkpoint = torch.load(
                model_file, map_location=device, weights_only=False
            )  # nosec B614 - Trusted model checkpoint, created by our training pipeline

        # Get required parameters for HybridConfig
        config_dict = checkpoint["config"]

        # Initialize HybridConfig with required parameters
        config = HybridConfig(
            model_type=config_dict.get(
                "model_type", "hybrid"
            ),  # Default to 'hybrid' if not specified
            n_users=test_dataset.n_users,  # Get from dataset
            n_items=test_dataset.n_items,  # Get from dataset
        )

        # Set remaining attributes from the dictionary
        for key, value in config_dict.items():
            if key not in [
                "model_type",
                "n_users",
                "n_items",
            ]:  # Skip already set attributes
                setattr(config, key, value)

        # Ensure device is set
        config.device = device

        # Initialize model with the reconstructed config
        model = HybridRecommender(config)
        model.load(model_file)

        model_loading.update(1)
        model_loading.close()

        # Generate predictions with progress bar
        logger.info("Starting evaluation...")
        all_predictions, all_targets = asyncio.run(
            evaluate_model_with_progress(model, test_data, device)
        )

        # Calculate metrics with progress
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(all_predictions, all_targets)

        # Save results with progress
        logger.info("Saving results...")
        saving = tqdm(total=2, desc="Saving results", position=0, leave=True)

        results_df = pd.DataFrame({"Actual": all_targets, "Predicted": all_predictions})
        results_df.to_csv(output_path / "predictions.csv", index=False)
        saving.update(1)

        metrics_df = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
        )
        metrics_df.to_csv(output_path / "evaluation_metrics.csv", index=False)
        saving.update(1)
        saving.close()

        logger.info("\nEvaluation completed successfully.")
        logger.info("\nMetrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise
