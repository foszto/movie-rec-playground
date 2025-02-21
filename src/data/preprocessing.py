#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path

class MovieLensPreprocessor:
    """Preprocess MovieLens dataset with detailed logging and error handling."""
    
    def __init__(self, min_user_ratings: int = 5, min_movie_ratings: int = 5, min_tags_per_item: int = 3):
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        self.min_tags_per_item = min_tags_per_item
        
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
    
    def preprocess(self,
                  ratings_df: pd.DataFrame,
                  movies_df: pd.DataFrame,
                  tags_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
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
            self.stats['initial_users'] = ratings_df['userId'].nunique()
            self.stats['initial_movies'] = movies_df['movieId'].nunique()
            self.stats['initial_ratings'] = len(ratings_df)
            
            # Ensure correct data types
            ratings_df = ratings_df.astype({
                'userId': np.int64,
                'movieId': np.int64,
                'rating': np.float32,
                'timestamp': np.int64
            })
            
            movies_df = movies_df.astype({
                'movieId': np.int64,
                'title': str,
                'genres': str
            })
            
            # Filter users and movies with minimum ratings
            self.logger.info("Filtering users and movies...")
            ratings_df = self._filter_minimum_ratings(ratings_df)
            
            # Keep only movies that have ratings
            active_movies = ratings_df['movieId'].unique()
            movies_df = movies_df[movies_df['movieId'].isin(active_movies)].copy()
            
            self.logger.info("Encoding user and movie IDs...")
            # Reset index to ensure continuous IDs
            ratings_df = ratings_df.reset_index(drop=True)
            movies_df = movies_df.reset_index(drop=True)
            
            # Convert IDs to consecutive integers
            ratings_df['userId'] = self.user_encoder.fit_transform(ratings_df['userId'])
            
            movie_ids = pd.concat([
                ratings_df['movieId'],
                movies_df['movieId']
            ]).unique()
            self.movie_encoder.fit(movie_ids)
            
            ratings_df['movieId'] = self.movie_encoder.transform(ratings_df['movieId'])
            movies_df['movieId'] = self.movie_encoder.transform(movies_df['movieId'])
            
            # Process genres
            self.logger.info("Processing genres...")
            movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
            
            # Add temporal features
            self.logger.info("Adding temporal features...")
            ratings_df = self._add_temporal_features(ratings_df)
            
            # Log final stats
            self.stats['final_users'] = ratings_df['userId'].nunique()
            self.stats['final_movies'] = movies_df['movieId'].nunique()
            self.stats['final_ratings'] = len(ratings_df)
            
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

    def preprocess_tags(self, tags_df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean tags data using batch processing."""
        self.logger.info("Processing tags data...")
        
        # Alapvető típuskonverzió és tisztítás
        tags_df = tags_df.astype({
            'userId': np.int64,
            'movieId': np.int64,
            'tag': str,
            'timestamp': np.int64
        })
        
        # Először szűrjük a tag-eket a valid user és movie ID-kra
        valid_user_ids = set(self.user_encoder.classes_)
        valid_movie_ids = set(self.movie_encoder.classes_)
        
        # Szűrés valid ID-kra
        tags_df = tags_df[
            (tags_df['userId'].isin(valid_user_ids)) & 
            (tags_df['movieId'].isin(valid_movie_ids))
        ]
        
        if len(tags_df) == 0:
            self.logger.warning("No valid tags found after filtering.")
            return pd.DataFrame(columns=['userId', 'movieId', 'tag', 'timestamp', 'weight'])
        
        # Tag tisztítás
        tags_df['tag'] = tags_df['tag'].str.lower().str.strip()
        initial_size = len(tags_df)
        
        # Ritka tagek szűrése
        self.logger.info("Filtering rare tags...")
        tag_counts = tags_df['tag'].value_counts()
        common_tags = tag_counts[tag_counts >= 2].index
        tags_df = tags_df[tags_df['tag'].isin(common_tags)]
        
        # Időbélyeg normalizálás
        self.logger.info("Normalizing timestamps...")
        min_timestamp = tags_df['timestamp'].min()
        max_timestamp = tags_df['timestamp'].max()
        timestamp_range = max_timestamp - min_timestamp
        tags_df['timestamp_norm'] = (tags_df['timestamp'] - min_timestamp) / timestamp_range if timestamp_range > 0 else 0
        
        # ID konverzió
        self.logger.info("Converting IDs...")
        try:
            tags_df['userId'] = self.user_encoder.transform(tags_df['userId'])
            tags_df['movieId'] = self.movie_encoder.transform(tags_df['movieId'])
        except ValueError as e:
            self.logger.error(f"Error during ID conversion: {str(e)}")
            # Debug információ
            invalid_users = set(tags_df['userId'].unique()) - valid_user_ids
            invalid_movies = set(tags_df['movieId'].unique()) - valid_movie_ids
            if invalid_users:
                self.logger.debug(f"Invalid user IDs found: {invalid_users}")
            if invalid_movies:
                self.logger.debug(f"Invalid movie IDs found: {invalid_movies}")
            raise
        
        # Batch feldolgozás
        self.logger.info("Processing tags in batches...")
        processed_dfs = []
        total_movies = len(tags_df['movieId'].unique())
        
        for idx, movie_id in enumerate(tags_df['movieId'].unique()):
            if idx % 100 == 0:
                self.logger.info(f"Processing movie {idx}/{total_movies}")
                
            movie_tags = tags_df[tags_df['movieId'] == movie_id].copy()
            
            if len(movie_tags) == 0:
                continue
                
            # Calculate importance
            tag_user_counts = movie_tags.groupby('tag')['userId'].nunique()
            movie_tags['importance'] = 1.0
            movie_tags.loc[movie_tags['tag'].isin(tag_user_counts[tag_user_counts >= 3].index), 'importance'] = 1.5
            
            # Calculate weight
            movie_tags['weight'] = movie_tags['importance'] * (1 + 0.5 * movie_tags['timestamp_norm'])
            
            # Keep only most relevant
            movie_tags = movie_tags.sort_values(['weight', 'timestamp'], ascending=[False, False])
            movie_tags = movie_tags.drop_duplicates(['userId', 'movieId', 'tag'], keep='first')
            
            processed_dfs.append(movie_tags[['userId', 'movieId', 'tag', 'timestamp', 'weight']])
        
        # Final combination
        if processed_dfs:
            tags_df = pd.concat(processed_dfs, ignore_index=True)
        else:
            tags_df = pd.DataFrame(columns=['userId', 'movieId', 'tag', 'timestamp', 'weight'])
        
        self.logger.info(f"Tag processing complete: {len(tags_df)} tags for {len(tags_df['movieId'].unique())} movies")
        return tags_df

    def create_tag_dictionaries(self, tags_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Create dictionaries for user and movie tags with optimized memory usage."""
        self.logger.info("Creating tag dictionaries...")
        
        # Initialize empty dictionaries
        user_tags = {}
        movie_tags = {}
        
        # Group by user and movie once
        user_groups = dict(tuple(tags_df.groupby('userId')))
        movie_groups = dict(tuple(tags_df.groupby('movieId')))
        
        # Create user dictionary
        total_users = len(user_groups)
        for i, (user_id, user_data) in enumerate(user_groups.items(), 1):
            if i % 1000 == 0:
                self.logger.info(f"Processing user tags: {i}/{total_users}")
            user_tags[user_id] = user_data[['movieId', 'tag', 'timestamp', 'weight']].copy()
        
        # Create movie dictionary
        total_movies = len(movie_groups)
        for i, (movie_id, movie_data) in enumerate(movie_groups.items(), 1):
            if i % 1000 == 0:
                self.logger.info(f"Processing movie tags: {i}/{total_movies}")
            movie_tags[movie_id] = movie_data[['userId', 'tag', 'timestamp', 'weight']].copy()
        
        self.logger.info(f"Created tag dictionaries for {len(user_tags)} users and {len(movie_tags)} movies")
        return user_tags, movie_tags

    def _filter_minimum_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and movies with minimum number of ratings."""
        self.logger.info(f"Filtering: minimum {self.min_user_ratings} ratings per user, "
                        f"{self.min_movie_ratings} ratings per movie")
        
        initial_shape = ratings_df.shape
        
        # Filter in steps for better logging
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        
        ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
        self.logger.info(f"After user filtering: {len(ratings_df)} ratings")
        
        movie_counts = ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index
        
        ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movies)]
        self.logger.info(f"After movie filtering: {len(ratings_df)} ratings")
        
        # Second pass to ensure all users still meet minimum after movie filtering
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
        
        self.logger.info(f"Reduced dataset from {initial_shape[0]} to {len(ratings_df)} ratings")
        return ratings_df
    
    def _add_temporal_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to ratings DataFrame."""
        timestamps = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        ratings_df['day_of_week'] = timestamps.dt.dayofweek
        ratings_df['hour_of_day'] = timestamps.dt.hour
        ratings_df['month'] = timestamps.dt.month
        ratings_df['year'] = timestamps.dt.year
        
        self.logger.debug("Added temporal features: day_of_week, hour_of_day, month, year")
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
        
        if self.stats['initial_ratings'] > 0:
            retention = (self.stats['final_ratings'] / self.stats['initial_ratings']) * 100
            self.logger.info(f"Data retention: {retention:.2f}%")
    
    def create_user_history_dict(self, 
                               ratings_df: pd.DataFrame,
                               movies_df: pd.DataFrame) -> Dict:
        """Create dictionary of user rating histories."""
        try:
            self.logger.info("Creating user history dictionary...")
            user_histories = {}
            
            # Merge ratings with movie information
            merged_df = ratings_df.merge(movies_df, on='movieId')
            
            for user_id in ratings_df['userId'].unique():
                user_ratings = merged_df[merged_df['userId'] == user_id].copy()
                user_histories[user_id] = user_ratings
                
                if len(user_histories) % 1000 == 0:
                    self.logger.debug(f"Processed {len(user_histories)} users")
            
            self.logger.info(f"Created histories for {len(user_histories)} users")
            return user_histories
            
        except Exception as e:
            self.logger.error(f"Error creating user history dict: {str(e)}", exc_info=True)
            raise
    
    def create_movie_info_dict(self, movies_df: pd.DataFrame) -> Dict:
        """Create dictionary of movie information."""
        try:
            self.logger.info("Creating movie info dictionary...")
            movie_info_dict = {
                row['movieId']: {
                    'movieId': row['movieId'],  
                    'title': row['title'],
                    'genres': row['genres'],
                    'description': ''  # Could be filled from external source
                }
                for _, row in movies_df.iterrows()
            }
            
            self.logger.info(f"Created info dict for {len(movie_info_dict)} movies")
            # Debug log for first movie
            if len(movie_info_dict) > 0:
                first_movie_id = list(movie_info_dict.keys())[0]
                self.logger.info(f"Sample movie info: {movie_info_dict[first_movie_id]}")
            
            return movie_info_dict
            
        except Exception as e:
            self.logger.error(f"Error creating movie info dict: {str(e)}", exc_info=True)
            raise