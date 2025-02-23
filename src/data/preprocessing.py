#!/usr/bin/env python3
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class MovieLensPreprocessor:
    """Preprocess MovieLens dataset with detailed logging and error handling."""

    def __init__(
        self,
        min_user_ratings: int = 5,
        min_movie_ratings: int = 5,
        min_tags_per_item: int = 3,
    ):
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        self.min_tags_per_item = min_tags_per_item
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Encoders for user and movie IDs
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

        # Stats for logging
        self.stats = {}

    def _log_dataframe_info(self, df: pd.DataFrame, name: str) -> None:
        """Log essential information about a DataFrame."""
        self.logger.info(f"\n{name} DataFrame Summary:")
        self.logger.info(f"Shape: {df.shape}")
        self.logger.info(f"Columns: {df.columns.tolist()}")
        self.logger.info("Data types:")
        for col, dtype in df.dtypes.items():
            self.logger.info(f"  {col}: {dtype}")
        self.logger.info("Missing values:")
        for col, missing in df.isnull().sum().items():
            if missing > 0:
                self.logger.info(f"  {col}: {missing}")
        # Csak az első 5 sor első pár oszlopát írjuk ki
        sample_cols = df.columns[:3]  # Csak az első 3 oszlop
        self.logger.info("\nSample data (first 5 rows, first 3 columns):")
        self.logger.info(f"\n{df[sample_cols].head().to_string()}")

    def preprocess(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        tags_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Preprocess the MovieLens dataset with detailed logging.

        Args:
            ratings_df: DataFrame with columns (userId, movieId, rating, timestamp)
            movies_df: DataFrame with columns (movieId, title, genres)
        """
        try:
            self.logger.info("Starting preprocessing...")

            # Log initial data info
            self._log_dataframe_info(ratings_df, "Initial Ratings")
            self._log_dataframe_info(movies_df, "Initial Movies")

            # Store initial stats
            self.stats["initial_users"] = ratings_df["userId"].nunique()
            self.stats["initial_movies"] = movies_df["movieId"].nunique()
            self.stats["initial_ratings"] = len(ratings_df)

            # Ensure correct data types
            ratings_df = ratings_df.astype(
                {
                    "userId": np.int64,
                    "movieId": np.int64,
                    "rating": np.float32,
                    "timestamp": np.int64,
                }
            )

            movies_df = movies_df.astype(
                {"movieId": np.int64, "title": str, "genres": str}
            )

            # Filter users and movies with minimum ratings
            self.logger.info("Filtering users and movies...")
            ratings_df = self._filter_minimum_ratings(ratings_df)

            # Keep only movies that have ratings
            active_movies = ratings_df["movieId"].unique()
            movies_df = movies_df[movies_df["movieId"].isin(active_movies)].copy()

            self.logger.info("Encoding user and movie IDs...")
            # Reset index to ensure continuous IDs
            ratings_df = ratings_df.reset_index(drop=True)
            movies_df = movies_df.reset_index(drop=True)

            # Convert IDs to consecutive integers
            ratings_df["userId"] = self.user_encoder.fit_transform(ratings_df["userId"])

            movie_ids = pd.concat(
                [ratings_df["movieId"], movies_df["movieId"]]
            ).unique()
            self.movie_encoder.fit(movie_ids)

            ratings_df["movieId"] = self.movie_encoder.transform(ratings_df["movieId"])
            movies_df["movieId"] = self.movie_encoder.transform(movies_df["movieId"])

            # Process genres
            self.logger.info("Processing genres...")
            movies_df["genres"] = movies_df["genres"].apply(lambda x: x.split("|"))

            # Add temporal features
            self.logger.info("Adding temporal features...")
            ratings_df = self._add_temporal_features(ratings_df)

            # Log final stats
            self.stats["final_users"] = ratings_df["userId"].nunique()
            self.stats["final_movies"] = movies_df["movieId"].nunique()
            self.stats["final_ratings"] = len(ratings_df)

            self._log_preprocessing_stats()

            # Final data check
            self._log_dataframe_info(ratings_df, "Final Ratings")
            self._log_dataframe_info(movies_df, "Final Movies")

            self.logger.info("Preprocessing completed successfully.")

            if tags_df is not None:
                tags_df = self.preprocess_tags(tags_df)
                return ratings_df, movies_df, tags_df

            return ratings_df, movies_df, None

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise

    def _parallel_process_tags(self, movie_tags: pd.DataFrame) -> pd.DataFrame:
        """Process tags for a single movie with memory efficiency."""
        if len(movie_tags) == 0:
            return pd.DataFrame()

        try:
            # Select only needed columns
            movie_tags = movie_tags[["userId", "movieId", "tag", "timestamp"]].copy()

            # Convert to GPU tensors for faster computation
            timestamps = torch.tensor(
                movie_tags["timestamp"].values, dtype=torch.float32
            ).to(self.device)

            # Normalize timestamps on GPU
            min_timestamp = timestamps.min()
            max_timestamp = timestamps.max()
            timestamp_range = max_timestamp - min_timestamp
            timestamp_norm = (
                (timestamps - min_timestamp) / timestamp_range
                if timestamp_range > 0
                else torch.zeros_like(timestamps)
            )

            # Calculate importance on CPU efficiently
            tag_user_counts = movie_tags.groupby("tag")["userId"].nunique()
            popular_tags = set(tag_user_counts[tag_user_counts >= 3].index)
            importance_array = np.ones(len(movie_tags))
            importance_array[movie_tags["tag"].isin(popular_tags)] = 1.5

            # Convert to GPU for weight calculation
            importance = torch.tensor(importance_array, dtype=torch.float32).to(
                self.device
            )
            weights = importance * (1 + 0.5 * timestamp_norm)

            # Move results back to CPU and update DataFrame
            movie_tags["timestamp_norm"] = timestamp_norm.cpu().numpy()
            movie_tags["importance"] = importance.cpu().numpy()
            movie_tags["weight"] = weights.cpu().numpy()

            # Clean up GPU memory
            del timestamps, importance, weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return movie_tags[["userId", "movieId", "tag", "timestamp", "weight"]]

        except Exception as e:
            self.logger.error(
                f"Error in _parallel_process_tags: {str(e)}", exc_info=True
            )
            raise

    def create_tag_dictionaries(self, tags_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Create dictionaries for user and movie tags with optimized memory usage."""
        try:
            from tqdm import tqdm

            self.logger.info("Creating tag dictionaries...")

            # Initialize empty dictionaries
            user_tags = {}
            movie_tags = {}

            # Group by user and movie once
            user_groups = dict(tuple(tags_df.groupby("userId")))
            movie_groups = dict(tuple(tags_df.groupby("movieId")))

            # Create user dictionary with progress bar
            with tqdm(
                total=len(user_groups), desc="Processing user tags", unit="users"
            ) as pbar:
                for user_id, user_data in user_groups.items():
                    user_tags[user_id] = user_data[
                        ["movieId", "tag", "timestamp", "weight"]
                    ].copy()
                    pbar.update(1)

            # Create movie dictionary with progress bar
            with tqdm(
                total=len(movie_groups), desc="Processing movie tags", unit="movies"
            ) as pbar:
                for movie_id, movie_data in movie_groups.items():
                    movie_tags[movie_id] = movie_data[
                        ["userId", "tag", "timestamp", "weight"]
                    ].copy()
                    pbar.update(1)

            self.logger.info(
                f"Created tag dictionaries for {len(user_tags)} users and {len(movie_tags)} movies"
            )
            return user_tags, movie_tags

        except Exception as e:
            self.logger.error(
                f"Error in create_tag_dictionaries: {str(e)}", exc_info=True
            )
            raise

    def preprocess_tags(self, tags_df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean tags data using GPU acceleration with progress bars."""
        try:
            from tqdm import tqdm

            self.logger.info("Processing tags data with GPU acceleration...")

            # Basic type conversion and cleaning
            tags_df = tags_df.astype(
                {
                    "userId": np.int64,
                    "movieId": np.int64,
                    "tag": str,
                    "timestamp": np.int64,
                }
            )

            # Filter tags for valid user and movie IDs
            valid_user_ids = set(self.user_encoder.classes_)
            valid_movie_ids = set(self.movie_encoder.classes_)

            tags_df = tags_df[
                (tags_df["userId"].isin(valid_user_ids))
                & (tags_df["movieId"].isin(valid_movie_ids))
            ]

            if len(tags_df) == 0:
                self.logger.warning("No valid tags found after filtering.")
                return pd.DataFrame(
                    columns=["userId", "movieId", "tag", "timestamp", "weight"]
                )

            # Clean tags
            tags_df["tag"] = tags_df["tag"].str.lower().str.strip()

            # Filter rare tags
            tag_counts = tags_df["tag"].value_counts()
            common_tags = tag_counts[tag_counts >= 2].index
            tags_df = tags_df[tags_df["tag"].isin(common_tags)]

            # Process each movie's tags in parallel using GPU
            processed_dfs = []

            # Process in batches for better GPU utilization
            batch_size = 100
            unique_movies = tags_df["movieId"].unique()
            movie_batches = np.array_split(
                unique_movies, max(1, len(unique_movies) // batch_size)
            )

            # Main processing loop with progress bar
            with tqdm(
                total=len(movie_batches), desc="Processing tag batches", unit="batch"
            ) as pbar:
                for movie_batch in movie_batches:
                    batch_tags = tags_df[tags_df["movieId"].isin(movie_batch)].copy()
                    processed_batch = []

                    # Inner loop for individual movies in batch
                    for movie_id in movie_batch:
                        movie_tags = batch_tags[
                            batch_tags["movieId"] == movie_id
                        ].copy()
                        processed_movie_tags = self._parallel_process_tags(movie_tags)
                        if len(processed_movie_tags) > 0:
                            processed_batch.append(processed_movie_tags)

                    if processed_batch:
                        processed_dfs.extend(processed_batch)
                    pbar.update(1)

            # Combine results
            if processed_dfs:
                tags_df = pd.concat(processed_dfs, ignore_index=True)
                tags_df = tags_df.sort_values(
                    ["movieId", "weight", "timestamp"], ascending=[True, False, False]
                )
                tags_df = tags_df.drop_duplicates(
                    ["userId", "movieId", "tag"], keep="first"
                )
            else:
                tags_df = pd.DataFrame(
                    columns=["userId", "movieId", "tag", "timestamp", "weight"]
                )

            self.logger.info(
                f"Tag processing complete: {len(tags_df)} tags for {len(tags_df['movieId'].unique())} movies"
            )
            return tags_df[["userId", "movieId", "tag", "timestamp", "weight"]]

        except Exception as e:
            self.logger.error(f"Error in preprocess_tags: {str(e)}", exc_info=True)
            raise

    def _filter_minimum_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and movies with minimum number of ratings."""
        self.logger.info(
            f"Filtering: minimum {self.min_user_ratings} ratings per user, "
            f"{self.min_movie_ratings} ratings per movie"
        )

        initial_shape = ratings_df.shape

        # Filter in steps for better logging
        user_counts = ratings_df["userId"].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index

        ratings_df = ratings_df[ratings_df["userId"].isin(valid_users)]
        self.logger.info(f"After user filtering: {len(ratings_df)} ratings")

        movie_counts = ratings_df["movieId"].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index

        ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movies)]
        self.logger.info(f"After movie filtering: {len(ratings_df)} ratings")

        # Second pass to ensure all users still meet minimum after movie filtering
        user_counts = ratings_df["userId"].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        ratings_df = ratings_df[ratings_df["userId"].isin(valid_users)]

        self.logger.info(
            f"Reduced dataset from {initial_shape[0]} to {len(ratings_df)} ratings"
        )
        return ratings_df

    def _add_temporal_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to ratings DataFrame."""
        timestamps = pd.to_datetime(ratings_df["timestamp"], unit="s")

        ratings_df["day_of_week"] = timestamps.dt.dayofweek
        ratings_df["hour_of_day"] = timestamps.dt.hour
        ratings_df["month"] = timestamps.dt.month
        ratings_df["year"] = timestamps.dt.year

        self.logger.debug(
            "Added temporal features: day_of_week, hour_of_day, month, year"
        )
        return ratings_df

    def _log_preprocessing_stats(self):
        """Log preprocessing statistics."""
        self.logger.info("\nPreprocessing Statistics:")
        self.logger.info(f"Initial number of users: {self.stats['initial_users']}")
        self.logger.info(f"Final number of users: {self.stats['final_users']}")
        self.logger.info(f"Initial number of movies: {self.stats['initial_movies']}")
        self.logger.info(f"Final number of movies: {self.stats['final_movies']}")
        self.logger.info(f"Initial number of ratings: {self.stats['initial_ratings']}")
        self.logger.info(f"Final number of ratings: {self.stats['final_ratings']}")

        if self.stats["initial_ratings"] > 0:
            retention = (
                self.stats["final_ratings"] / self.stats["initial_ratings"]
            ) * 100
            self.logger.info(f"Data retention: {retention:.2f}%")

    def create_movie_info_dict(self, movies_df: pd.DataFrame) -> Dict:
        """Create dictionary of movie information with memory-efficient processing."""
        try:
            import gc

            from tqdm import tqdm

            self.logger.info("Creating movie info dictionary...")

            # Select only needed columns
            movies_df = movies_df[["movieId", "title", "genres"]].copy()
            movie_info_dict = {}

            # Process in batches
            batch_size = 1000
            total_movies = len(movies_df)

            with tqdm(
                total=total_movies, desc="Processing movies", unit="movie"
            ) as pbar:
                for start_idx in range(0, total_movies, batch_size):
                    end_idx = min(start_idx + batch_size, total_movies)
                    batch_df = movies_df.iloc[start_idx:end_idx]

                    for _, row in batch_df.iterrows():
                        movie_info_dict[row["movieId"]] = {
                            "movieId": row["movieId"],
                            "title": row["title"],
                            "genres": row["genres"],
                        }
                        pbar.update(1)

                    # Clean up batch memory
                    del batch_df
                    gc.collect()

            self.logger.info(f"Created info dict for {len(movie_info_dict)} movies")
            return movie_info_dict

        except Exception as e:
            self.logger.error(
                f"Error in create_movie_info_dict: {str(e)}", exc_info=True
            )
            raise

    def create_user_history_dict(
        self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
    ) -> Dict:
        """Create dictionary of user rating histories with memory-efficient processing."""
        try:
            import gc

            self.logger.info("Creating user history dictionary...")

            # Optimize memory by selecting only needed columns
            ratings_df = ratings_df[["userId", "movieId", "rating", "timestamp"]]
            movies_df = movies_df[["movieId", "title", "genres"]]

            unique_users = ratings_df["userId"].unique()
            total_users = len(unique_users)
            batch_size = min(100, max(1, total_users // 20))

            user_histories = {}
            processed_count = 0

            for start_idx in range(0, total_users, batch_size):
                # Process users in batches
                end_idx = min(start_idx + batch_size, total_users)
                batch_users = unique_users[start_idx:end_idx]

                # Create batch only for current users
                batch_ratings = ratings_df[ratings_df["userId"].isin(batch_users)]

                # Process each user in the batch
                for user_id in batch_users:
                    user_ratings = batch_ratings[batch_ratings["userId"] == user_id]
                    user_history = user_ratings.merge(
                        movies_df, on="movieId", how="left"
                    )

                    # Store only necessary columns
                    user_histories[user_id] = user_history[
                        ["movieId", "rating", "timestamp", "title", "genres"]
                    ]

                    processed_count += 1

                    # Log progress every 1000 users
                    if processed_count % 1000 == 0:
                        self.logger.info(
                            f"Processed {processed_count}/{total_users} users "
                            f"({processed_count/total_users*100:.1f}%)"
                        )

                # Clear batch data and force garbage collection
                del batch_ratings
                gc.collect()

            self.logger.info(f"Created histories for {len(user_histories)} users")
            return user_histories

        except Exception as e:
            self.logger.error(
                f"Error in create_user_history_dict: {str(e)}", exc_info=True
            )
            raise
