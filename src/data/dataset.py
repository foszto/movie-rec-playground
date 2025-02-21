import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class MovieLensDataset(Dataset):
    """Dataset class for MovieLens data."""
    
    def __init__(self, 
                 ratings_df: pd.DataFrame,
                 user_map: Dict[int, int] = None,
                 item_map: Dict[int, int] = None):
        """
        Initialize MovieLens dataset.
        
        Args:
            ratings_df: DataFrame with columns (userId, movieId, rating, timestamp)
            user_map: Mapping from original to consecutive user IDs
            item_map: Mapping from original to consecutive movie IDs
        """
        self.ratings_df = ratings_df
        
        # Create ID mappings if not provided
        self.user_map = user_map or self._create_id_mapping(ratings_df['userId'])
        self.item_map = item_map or self._create_id_mapping(ratings_df['movieId'])
        
        # Convert IDs to tensor indices
        self.users = torch.tensor([self.user_map[uid] for uid in ratings_df['userId']], 
                                dtype=torch.long)
        self.items = torch.tensor([self.item_map[iid] for iid in ratings_df['movieId']], 
                                dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
        
        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
    
    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'user_id': self.users[idx],
            'item_id': self.items[idx],
            'rating': self.ratings[idx]
        }
    
    @staticmethod
    def _create_id_mapping(ids: pd.Series) -> Dict[int, int]:
        """Create a mapping from original IDs to consecutive indices."""
        unique_ids = sorted(ids.unique())
        return {original: idx for idx, original in enumerate(unique_ids)}
    
    @classmethod
    def train_valid_split(cls, 
                         ratings_df: pd.DataFrame,
                         valid_ratio: float = 0.2,
                         random_seed: Optional[int] = None
                         ) -> Tuple['MovieLensDataset', 'MovieLensDataset']:
        """Split data into training and validation sets."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create ID mappings using all data
        user_map = cls._create_id_mapping(ratings_df['userId'])
        item_map = cls._create_id_mapping(ratings_df['movieId'])
        
        # Split by users to prevent leakage
        users = ratings_df['userId'].unique()
        valid_users = np.random.choice(
            users,
            size=int(len(users) * valid_ratio),
            replace=False
        )
        
        valid_mask = ratings_df['userId'].isin(valid_users)
        train_df = ratings_df[~valid_mask]
        valid_df = ratings_df[valid_mask]
        
        return (
            cls(train_df, user_map, item_map),
            cls(valid_df, user_map, item_map)
        )
    
    def get_dataloader(self, 
                      batch_size: int,
                      shuffle: bool = True
                      ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )