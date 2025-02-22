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
from src.configs.model_config import HybridConfig
from torch.serialization import add_safe_globals, safe_globals

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Directory containing test data')
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Directory to save evaluation results')
def evaluate(data_dir: str, model_path: str, output_dir: str):
    """Evaluate the trained model."""
    
    setup_logging()
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        logger.info("Loading test data...")
        (ratings_df, movies_df, tags_df, 
         user_history_dict, movie_info_dict,
         user_tags_dict, movie_tags_dict) = load_preprocessed_data(data_dir)

        # Create test dataset
        test_dataset = MovieLensDataset(ratings_df)
        test_data = {
            'dataloader': test_dataset.get_dataloader(batch_size=64, shuffle=False),
            'movie_info_dict': movie_info_dict,
            'user_history_dict': user_history_dict,
            'user_tags_dict': user_tags_dict,
            'movie_tags_dict': movie_tags_dict
        }

        # Load model
        logger.info("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_file = list(Path(model_path).glob('*.pt'))[0]  # Find the .pt file
        
        # Add HybridConfig to safe globals for loading
        add_safe_globals([HybridConfig])
        
        # Try loading with different options if needed
        try:
            # First try with weights_only=True
            checkpoint = torch.load(model_file, map_location=device, weights_only=True)
        except Exception as e:
            logger.warning(f"Could not load with weights_only=True, trying with weights_only=False")
            # If that fails, try with weights_only=False
            with safe_globals([HybridConfig]):
                checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        
        # Initialize and load the model
        model = HybridRecommender(checkpoint['config'])
        model.load(model_file)
        
        # Generate predictions
        logger.info("Generating predictions...")
        async def evaluate_model():
            all_predictions = []
            all_targets = []

            with torch.no_grad():  # Disable gradient calculation
                for batch in test_data['dataloader']:
                    user_ids = batch['user_id'].to(device)
                    item_ids = batch['item_id'].to(device)
                    ratings = batch['rating']

                    predictions = await model.predict(
                        user_ids,
                        item_ids,
                        test_data['movie_info_dict'],
                        test_data['user_history_dict']
                    )

                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(ratings.numpy())
            
            return np.array(all_predictions), np.array(all_targets)

        # Run evaluation
        all_predictions, all_targets = asyncio.run(evaluate_model())

        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)

        # Save results
        results_df = pd.DataFrame({
            'Actual': all_targets,
            'Predicted': all_predictions
        })
        results_df.to_csv(output_path / 'predictions.csv', index=False)

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
    evaluate()