# src/cli/train.py
import click
import logging
import yaml
from pathlib import Path
import asyncio
from src.configs.logging_config import setup_logging
from src.configs.model_config import HybridConfig
from src.data.dataset import MovieLensDataset
from src.models.hybrid import HybridRecommender
from src.utils.data_io import load_preprocessed_data
from src.utils.visualization import plot_training_history

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Directory containing preprocessed data')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Directory to save model and results')
@click.option('--config-path', type=click.Path(exists=True), required=True,
              help='Path to model configuration file')
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
        ratings_df, movies_df, user_history_dict, movie_info_dict = load_preprocessed_data(data_dir)
        
        train_dataset, valid_dataset = MovieLensDataset.train_valid_split(
            ratings_df,
            valid_ratio=config.get('valid_ratio', 0.2),
            random_seed=config.get('random_seed', 42)
        )
        
        model_config = HybridConfig(
            model_type='hybrid_llm',
            n_users=train_dataset.n_users,
            n_items=train_dataset.n_items,
            **config.get('model_params', {})
        )
        
        model = HybridRecommender(model_config)
        
        train_data = {
            'dataloader': train_dataset.get_dataloader(batch_size=config.get('batch_size', 64)),
            'movie_info_dict': movie_info_dict,
            'user_history_dict': user_history_dict
        }
        valid_data = {
            'dataloader': valid_dataset.get_dataloader(batch_size=config.get('batch_size', 64)),
            'movie_info_dict': movie_info_dict,
            'user_history_dict': user_history_dict
        }
        
        logger.info("Starting training...")
        history = []
        best_valid_loss = float('inf')
        patience = config.get('early_stopping_patience', 5)
        patience_counter = 0
        
        for epoch in range(config.get('n_epochs', 10)):
            metrics = await model.fit(train_data, valid_data)
            history.append(metrics)
            
            valid_loss = metrics.get('valid_loss', float('inf'))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                model.save(output_path / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        model.save(output_path / 'final_model.pt')
        plot_training_history(history, save_path=output_path / 'training_history.png')
        
        logger.info("Training completed successfully.")
    
    asyncio.run(async_train())