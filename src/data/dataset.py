from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MovieLensDataset(Dataset):
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        user_map: Dict[int, int] = None,
        item_map: Dict[int, int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize MovieLens dataset with proper memory handling.
        """
        self.ratings_df = ratings_df
        self.device = device

        # Create ID mappings if not provided
        self.user_map = user_map or self._create_id_mapping(ratings_df["userId"])
        self.item_map = item_map or self._create_id_mapping(ratings_df["movieId"])

        # Always keep tensors on CPU initially
        self.users = torch.tensor(
            [self.user_map[uid] for uid in ratings_df["userId"]], dtype=torch.long
        )
        self.items = torch.tensor(
            [self.item_map[iid] for iid in ratings_df["movieId"]], dtype=torch.long
        )
        self.ratings = torch.tensor(ratings_df["rating"].values, dtype=torch.float)

        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Always return CPU tensors, let DataLoader handle device movement
        return {
            "user_id": self.users[idx],
            "item_id": self.items[idx],
            "rating": self.ratings[idx],
        }

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader with optimized memory handling.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to maintain worker processes between iterations
            **kwargs: Additional arguments to pass to DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers if num_workers > 0 else False,
            **kwargs
        )

    @staticmethod
    def _create_id_mapping(ids: pd.Series) -> Dict[int, int]:
        """Create a mapping from original IDs to consecutive indices."""
        unique_ids = sorted(ids.unique())
        return {original: idx for idx, original in enumerate(unique_ids)}

    @classmethod
    def train_valid_split(
        cls,
        ratings_df: pd.DataFrame,
        valid_ratio: float = 0.2,
        random_seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Tuple["MovieLensDataset", "MovieLensDataset"]:
        """Split data into training and validation sets."""
        if random_seed is not None:
            np.random.seed(random_seed)

        # Create ID mappings using all data
        user_map = cls._create_id_mapping(ratings_df["userId"])
        item_map = cls._create_id_mapping(ratings_df["movieId"])

        # Split by users to prevent leakage
        users = ratings_df["userId"].unique()
        valid_users = np.random.choice(
            users, size=int(len(users) * valid_ratio), replace=False
        )

        valid_mask = ratings_df["userId"].isin(valid_users)
        train_df = ratings_df[~valid_mask]
        valid_df = ratings_df[valid_mask]

        return (
            cls(train_df, user_map, item_map, device),
            cls(valid_df, user_map, item_map, device),
        )
