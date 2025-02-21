# src/utils/data_io.py
import pandas as pd
from pathlib import Path
import logging

import torch
from src.utils.serialization import load_movie_info, load_user_history, save_movie_info, save_user_history

def load_raw_data(data_dir: str):
    """Load raw MovieLens data from CSV files."""
    data_path = Path(data_dir)
    ratings_df = pd.read_csv(data_path / 'ratings.csv')
    movies_df = pd.read_csv(data_path / 'movies.csv')
    return ratings_df, movies_df

def load_preprocessed_data(data_dir: str):
    """Load preprocessed MovieLens data."""
    data_path = Path(data_dir)
    logger = logging.getLogger(__name__)
    try:
        ratings_df = pd.read_parquet(data_path / 'ratings.parquet')
        movies_df = pd.read_parquet(data_path / 'movies.parquet')
    except:
        ratings_df = pd.read_csv(data_path / 'ratings.csv')
        movies_df = pd.read_csv(data_path / 'movies.csv')

    try:
        # Ha .pt fájlokban vannak a szerializált adatok
        user_history_dict = torch.load(data_path / 'user_history.pt')
        movie_info_dict = torch.load(data_path / 'movie_info.pt')
    except:
        # Visszaesés a .pkl formátumra
        user_history_dict = load_user_history(data_path / 'user_history.pkl')
        movie_info_dict = load_movie_info(data_path / 'movie_info.pkl')

    return ratings_df, movies_df, user_history_dict, movie_info_dict
def save_processed_data(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                      user_history_dict: dict, movie_info_dict: dict, output_path: Path):
    """Save preprocessed data to disk."""
    logger = logging.getLogger(__name__)
    try:
        ratings_df.to_parquet(output_path / 'ratings.parquet')
        movies_df.to_parquet(output_path / 'movies.parquet')
    except ImportError:
        logger.warning("Parquet support not available, falling back to CSV format")
        ratings_df.to_csv(output_path / 'ratings.csv', index=False)
        movies_df.to_csv(output_path / 'movies.csv', index=False)
    
    save_user_history(user_history_dict, output_path / 'user_history.pkl')
    save_movie_info(movie_info_dict, output_path / 'movie_info.pkl')
