# src/cli/evaluate.py
import click
import logging
import pandas as pd
import torch
import numpy as np
import asyncio
from pathlib import Path
from src.configs.logging_config import setup_logging
from src.data.dataset import MovieLensDataset
from src.models.hybrid import HybridRecommender
from src.utils.data_io import load_preprocessed_data
from src.utils.metrics import calculate_metrics

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Directory containing test data')
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Directory to save evaluation results')
async def evaluate(data_dir: str, model_path: str, output_dir: str):
    """Evaluate the trained model."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        logger.info("Loading test data...")
        ratings_df, _, user_history_dict, movie_info_dict = load_preprocessed_data(data_dir)

        # Create test dataset
        test_dataset = MovieLensDataset(ratings_df)
        test_data = {
            'dataloader': test_dataset.get_dataloader(batch_size=128, shuffle=False),
            'movie_info_dict': movie_info_dict,
            'user_history_dict': user_history_dict
        }

        # Load model
        logger.info("Loading model...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = HybridRecommender(checkpoint['config'])
        model.load(model_path)

        # Generate predictions
        logger.info("Generating predictions...")
        all_predictions = []
        all_targets = []

        for batch in test_data['dataloader']:
            user_ids = batch['user_id']
            item_ids = batch['item_id']
            ratings = batch['rating']

            predictions = await model.predict(
                user_ids,
                item_ids,
                test_data['movie_info_dict'],
                test_data['user_history_dict']
            )

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.numpy())

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        metrics = calculate_metrics(all_predictions, all_targets)

        # Save detailed results
        results_df = pd.DataFrame({
            'Actual': all_targets,
            'Predicted': all_predictions
        })
        results_df.to_csv(output_path / 'predictions.csv', index=False)

        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        metrics_df.to_csv(output_path / 'evaluation_metrics.csv', index=False)

        logger.info("Evaluation completed successfully.")
        logger.info("Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    asyncio.run(evaluate())