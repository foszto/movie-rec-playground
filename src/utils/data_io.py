# src/utils/data_io.py
from typing import Optional
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
    tags_df = pd.read_csv(data_path / 'tags.csv')
    return ratings_df, movies_df, tags_df

def save_processed_data(ratings_df: pd.DataFrame, 
                      movies_df: pd.DataFrame,
                      tags_df: Optional[pd.DataFrame],
                      user_history_dict: dict, 
                      movie_info_dict: dict,
                      user_tags_dict: Optional[dict],
                      movie_tags_dict: Optional[dict],
                      output_path: Path):
    """Save preprocessed data to disk."""
    logger = logging.getLogger(__name__)
    try:
        ratings_df.to_parquet(output_path / 'ratings.parquet')
        movies_df.to_parquet(output_path / 'movies.parquet')
        if tags_df is not None:
            tags_df.to_parquet(output_path / 'tags.parquet')
    except ImportError:
        logger.warning("Parquet support not available, falling back to CSV format")
        ratings_df.to_csv(output_path / 'ratings.csv', index=False)
        movies_df.to_csv(output_path / 'movies.csv', index=False)
        if tags_df is not None:
            tags_df.to_csv(output_path / 'tags.csv', index=False)
    
    save_user_history(user_history_dict, output_path / 'user_history.pkl')
    save_movie_info(movie_info_dict, output_path / 'movie_info.pkl')
    
    if user_tags_dict is not None:
        torch.save(user_tags_dict, output_path / 'user_tags.pt')
    if movie_tags_dict is not None:
        torch.save(movie_tags_dict, output_path / 'movie_tags.pt')

def load_preprocessed_data(data_dir: str):
    """Load preprocessed MovieLens data."""
    data_path = Path(data_dir)
    logger = logging.getLogger(__name__)
    try:
        ratings_df = pd.read_parquet(data_path / 'ratings.parquet')
        movies_df = pd.read_parquet(data_path / 'movies.parquet')
        tags_df = pd.read_parquet(data_path / 'tags.parquet') if (data_path / 'tags.parquet').exists() else None
    except:
        ratings_df = pd.read_csv(data_path / 'ratings.csv')
        movies_df = pd.read_csv(data_path / 'movies.csv')
        tags_df = pd.read_csv(data_path / 'tags.csv') if (data_path / 'tags.csv').exists() else None

    try:
        user_history_dict = torch.load(data_path / 'user_history.pt')
        movie_info_dict = torch.load(data_path / 'movie_info.pt')
        user_tags_dict = torch.load(data_path / 'user_tags.pt') if (data_path / 'user_tags.pt').exists() else None
        movie_tags_dict = torch.load(data_path / 'movie_tags.pt') if (data_path / 'movie_tags.pt').exists() else None
    except:
        user_history_dict = load_user_history(data_path / 'user_history.pkl')
        movie_info_dict = load_movie_info(data_path / 'movie_info.pkl')
        user_tags_dict = None
        movie_tags_dict = None

    return ratings_df, movies_df, tags_df, user_history_dict, movie_info_dict, user_tags_dict, movie_tags_dict
