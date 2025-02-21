# src/llm/feature_extractor.py

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class LLMFeatureExtractor:
    """Extract features from movie descriptions and user preferences using LLM."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 128,
                 device: str = None):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Name of the sentence transformer model
            embedding_dim: Dimension of embeddings
            device: Device to use for computations
        """
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        self.model = SentenceTransformer(model_name)
        if device is not None:
            self.model = self.model.to(device)
        
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_cache = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_user_preference_embedding(self, user_history: pd.DataFrame) -> torch.Tensor:
        """
        Generate user embedding from rating history.
        
        Args:
            user_history: DataFrame with user's rating history
        Returns:
            torch.Tensor: User preference embedding
        """
        user_id = user_history['userId'].iloc[0]
        cache_key = f"user_{user_id}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Create user profile text
        top_movies = user_history[user_history['rating'] >= 4.0]['title'].tolist()[:5]
        genres = []
        for genre_list in user_history[user_history['rating'] >= 4.0]['genres']:
            if isinstance(genre_list, list):
                genres.extend(genre_list)
        favorite_genres = pd.Series(genres).value_counts().head(3).index.tolist()
        
        profile_text = f"""
        User preferences:
        Top rated movies: {', '.join(top_movies) if top_movies else 'None'}
        Favorite genres: {', '.join(favorite_genres) if favorite_genres else 'None'}
        Total ratings: {len(user_history)}
        Average rating: {user_history['rating'].mean():.2f}
        """
        
        # Get embedding
        try:
            embedding = self.model.encode(profile_text, convert_to_tensor=True)
            embedding = F.normalize(embedding, p=2, dim=0)
            
            if self.device:
                embedding = embedding.to(self.device)
            
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating user embedding: {str(e)}")
            return torch.randn(self.embedding_dim, device=self.device)
    
    async def get_movie_embedding(self, movie_info: Dict) -> torch.Tensor:
        """
        Generate movie embedding from movie information.
        
        Args:
            movie_info: Dictionary containing movie information
        Returns:
            torch.Tensor: Movie embedding
        """
        # Try different possible keys for movie ID
        movie_id = None
        for key in ['movieId', 'movie_id', 'id']:
            if key in movie_info:
                movie_id = movie_info[key]
                break
        
        if movie_id is None:
            # Ha nem találtunk ID-t, használjuk a szótár első kulcsát
            movie_id = list(movie_info.keys())[0]
            self.logger.warning(f"No standard movie ID found, using key: {movie_id}")
        
        cache_key = f"movie_{movie_id}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Create movie description text
        title = movie_info.get('title', 'Unknown Title')
        genres = movie_info.get('genres', [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split('|')]
        elif not isinstance(genres, list):
            genres = []
            
        description = movie_info.get('description', '')
        
        movie_text = f"""
        Title: {title}
        Genres: {', '.join(genres)}
        Description: {description}
        """
        
        # Get embedding
        try:
            embedding = self.model.encode(movie_text, convert_to_tensor=True)
            embedding = F.normalize(embedding, p=2, dim=0)
            
            if self.device:
                embedding = embedding.to(self.device)
            
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating movie embedding: {str(e)}")
            return torch.randn(self.embedding_dim, device=self.device)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")