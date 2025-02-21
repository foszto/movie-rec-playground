# utils/serialization.py
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any
import pickle
import logging

logger = logging.getLogger(__name__)

def save_user_history(user_history_dict: Dict, file_path: str) -> None:
    """
    Save user history dictionary in a safe format.
    """
    # Convert DataFrame to dictionary format
    serializable_dict = {}
    for user_id, df in user_history_dict.items():
        serializable_dict[user_id] = df.to_dict('records')
    
    with open(file_path, 'wb') as f:
        pickle.dump(serializable_dict, f)

def load_user_history(file_path: str) -> Dict:
    """
    Load user history dictionary safely.
    """
    with open(file_path, 'rb') as f:
        serializable_dict = pickle.load(f)
    
    # Convert back to DataFrame format
    user_history_dict = {}
    for user_id, records in serializable_dict.items():
        user_history_dict[user_id] = pd.DataFrame.from_records(records)
    
    return user_history_dict

def save_movie_info(movie_info_dict: Dict, file_path: str) -> None:
    """
    Save movie information dictionary.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(movie_info_dict, f)

def load_movie_info(file_path: str) -> Dict:
    """
    Load movie information dictionary.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)