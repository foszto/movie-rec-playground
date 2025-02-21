# src/cli/preprocess.py
import click
import logging
from pathlib import Path
import pandas as pd
from src.configs.logging_config import setup_logging
from src.data.preprocessing import MovieLensPreprocessor
from src.utils.data_io import save_processed_data, load_raw_data
from src.utils.visualization import plot_rating_distribution

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Directory containing MovieLens dataset files')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Directory to save preprocessed data')
@click.option('--min-user-ratings', type=int, default=5,
              help='Minimum number of ratings per user')
@click.option('--min-movie-ratings', type=int, default=5,
              help='Minimum number of ratings per movie')
@click.option('--min-tags-per-item', type=int, default=3,
              help='Minimum number of tags per item')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Logging level')
@click.option('--log-file', type=click.Path(), default=None,
              help='Optional log file path')
def preprocess(data_dir: str, output_dir: str, min_user_ratings: int, 
               min_movie_ratings: int, min_tags_per_item: int,
               log_level: str, log_file: str):
    """Preprocess MovieLens dataset."""
    setup_logging(
        log_level=log_level,
        log_file=log_file if log_file else None,
        clear_handlers=True
    )
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Loading data...")
        ratings_df, movies_df, tags_df = load_raw_data(data_dir)
        
        preprocessor = MovieLensPreprocessor(
            min_user_ratings=min_user_ratings,
            min_movie_ratings=min_movie_ratings,
            min_tags_per_item=min_tags_per_item
        )
        
        # Preprocess main data
        ratings_df, movies_df, tags_df = preprocessor.preprocess(ratings_df, movies_df, tags_df)
        
        # Create dictionaries
        user_history_dict = preprocessor.create_user_history_dict(ratings_df, movies_df)
        movie_info_dict = preprocessor.create_movie_info_dict(movies_df)
        
        # Create tag dictionaries if tags are available
        user_tags_dict = None
        movie_tags_dict = None
        if tags_df is not None:
            user_tags_dict, movie_tags_dict = preprocessor.create_tag_dictionaries(tags_df)
        
        logger.info("Saving preprocessed data...")
        save_processed_data(
            ratings_df, movies_df, tags_df,
            user_history_dict, movie_info_dict,
            user_tags_dict, movie_tags_dict,
            output_path
        )
        
        # Generate visualizations
        plot_rating_distribution(
            ratings_df['rating'], 
            save_path=output_path / 'rating_distribution.png'
        )
        
        #if tags_df is not None:
        #    plot_tag_distribution(
        #        tags_df['tag'].value_counts(),
        #        save_path=output_path / 'tag_distribution.png'
        #    )
        
        logger.info("Preprocessing completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise