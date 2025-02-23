# src/llm/feature_extractor.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from src.llm.genre_feature_processor import OptimizedGenreProcessor

class LLMFeatureExtractor:
    """Extract features from movie descriptions and user preferences using LLM."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 64,
                 native_dim = 384,
                 device: str = None):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Name of the sentence transformer model
            embedding_dim: Dimension of embeddings
            device: Device to use for computations
        """
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        self.model = SentenceTransformer(model_name, device=device)
        self.native_dim = native_dim
        self.embedding_dim = embedding_dim
        self.device = device

        self.projection = nn.Linear(self.native_dim, self.embedding_dim).to(self.device)
        if device is not None:
            self.model = self.model.to(device)
        
        self.embedding_dim = embedding_dim
        self.embedding_cache = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.tag_embedding = nn.Linear(native_dim, embedding_dim).to(device)
        self.tag_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        ).to(device)

        self.genre_processor = OptimizedGenreProcessor()
    
    async def process_user_tags(self, user_tags: pd.DataFrame) -> torch.Tensor:
        """Process user tags with enhanced genre focus."""
        self.logger.info("Processing user tags...")
        torch.set_grad_enabled(False)
        torch.set_default_device(self.device)
        
        if user_tags.empty:
            return torch.zeros(self.embedding_dim, device=self.device)
            
        # Weight tags based on their relevance to genres
        genre_related_tags = {
            'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
            'drama', 'family', 'fantasy', 'horror', 'mystery', 'romance', 'sci-fi',
            'thriller', 'war', 'western'
        }
        
        # Add weight column based on genre relevance
        user_tags['weight'] = user_tags['tag'].apply(
            lambda x: 2.0 if x.lower() in genre_related_tags else 1.0
        )
        
        # Aggregate tags by weighted frequency
        tag_weights = user_tags.groupby('tag')['weight'].sum()
        top_tags = tag_weights.nlargest(15).index.tolist()  # Increased from 10 to 15
        
        # Get embeddings for top tags
        tag_embeddings = []
        weights = []
        for tag in top_tags:
            embedding = self.model.encode(tag, convert_to_tensor=True).to(self.device)
            tag_embeddings.append(embedding)
            weights.append(tag_weights[tag])
            
        if not tag_embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)
            
        # Stack and process tag embeddings with emphasis on genre-related tags
        tag_tensor = torch.stack(tag_embeddings).to(self.device)
        tag_weights = torch.softmax(torch.tensor(weights), dim=0).to(self.device)
        
        # Project to lower dimension
        projected_tags = self.tag_embedding(tag_tensor)
        
        # Apply attention
        attended_tags, _ = self.tag_attention(
            projected_tags.unsqueeze(0),
            projected_tags.unsqueeze(0),
            projected_tags.unsqueeze(0)
        )
        
        # Weighted sum of tag embeddings
        tag_profile = (attended_tags.squeeze(0) * tag_weights.unsqueeze(1)).sum(0)
        return F.normalize(tag_profile, p=2, dim=0)
    
    async def process_movie_tags(self, movie_tags: pd.DataFrame) -> torch.Tensor:
        """Process movie tags to create tag-based movie profile."""
        self.logger.info("Processing movie tags...")
        if movie_tags.empty:
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # Aggregate tags by frequency and recency
        movie_tags['weight'] = 1.0
        movie_tags.loc[movie_tags['timestamp'] > movie_tags['timestamp'].median(), 'weight'] = 1.2
        
        tag_weights = movie_tags.groupby('tag')['weight'].sum()
        top_tags = tag_weights.nlargest(10).index.tolist()
        
        # Get embeddings for top tags
        tag_embeddings = []
        for tag in top_tags:
            embedding = self.model.encode(tag, convert_to_tensor=True)
            tag_embeddings.append(embedding)
            
        if not tag_embeddings:
            return torch.zeros(self.embedding_dim, device=self.device)
            
        # Stack and process tag embeddings
        tag_tensor = torch.stack(tag_embeddings)
        tag_importance = torch.softmax(torch.tensor(tag_weights[:len(tag_embeddings)]), dim=0)
        
        # Project to lower dimension
        projected_tags = self.tag_embedding(tag_tensor)
        
        # Apply attention
        attended_tags, _ = self.tag_attention(
            projected_tags.unsqueeze(0),
            projected_tags.unsqueeze(0),
            projected_tags.unsqueeze(0)
        )
        
        # Weighted sum of tag embeddings
        tag_profile = (attended_tags.squeeze(0) * tag_importance.unsqueeze(1)).sum(0)
        return F.normalize(tag_profile, p=2, dim=0)

    async def get_user_preference_embedding(self, 
                                         user_history: pd.DataFrame,
                                         user_tags: Optional[pd.DataFrame] = None) -> torch.Tensor:
        """Enhanced user embedding with tag information."""
        # Get base user embedding
        base_embedding = await self._get_base_user_embedding(user_history)
        
        # Get tag-based embedding if available
        if user_tags is not None and not user_tags.empty:
            tag_embedding = await self.process_user_tags(user_tags)
            # Combine embeddings with equal weights
            combined = (base_embedding * 0.7 + tag_embedding * 0.3) / 2
            return F.normalize(combined, p=2, dim=0)
        
        return base_embedding
    
    async def _get_base_user_embedding(self, user_history: pd.DataFrame) -> torch.Tensor:
        """Generate base user embedding with enhanced genre focus."""
        # Check cache
        cache_key = f"user_{len(user_history)}_{user_history['timestamp'].max()}_{user_history['rating'].mean()}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Weight movies based on ratings
        user_history['weight'] = user_history['rating'].apply(
            lambda x: 2.0 if x >= 4.0 else (0.5 if x <= 2.0 else 1.0)
        )
        
        # Get genre preferences with weighted history
        genre_preferences = self.genre_processor.get_genre_preferences(user_history)
        
        # Generate detailed genre text with emphasis on highly rated genres
        top_genres = sorted(
            [(g, genre_preferences['genre_scores'][g]) 
             for g in genre_preferences['liked_genres']],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Create detailed genre descriptions
        genre_descriptions = []
        for genre, score in top_genres:
            similar_genres = genre_preferences['similar_genres'].get(genre, [])
            similar_text = f" (similar to {', '.join(g['genre'] for g in similar_genres[:2])})" if similar_genres else ""
            genre_descriptions.append(
                f"Strong preference for {genre} (score: {score:.2f}){similar_text}"
            )
        
        # Calculate recent activity using timestamp as integer
        max_timestamp = user_history['timestamp'].max()
        # 90 days in seconds: 90 * 24 * 60 * 60 = 7776000
        recent_threshold = max_timestamp - 7776000
        recent_ratings = user_history[user_history['timestamp'] > recent_threshold]
        recent_high_ratings = recent_ratings[recent_ratings['rating'] >= 4.0]
        
        # Generate comprehensive profile text
        profile_text = f"""
        User Genre Preferences:
        {chr(10).join(genre_descriptions)}
        
        Recent Activity (last 90 days):
        Total ratings: {len(recent_ratings)}
        High ratings (4+): {len(recent_high_ratings)}
        Average rating: {recent_ratings['rating'].mean():.2f}
        
        Overall Stats:
        Total movies rated: {len(user_history)}
        Average rating: {user_history['rating'].mean():.2f}
        Rating spread: {user_history['rating'].std():.2f}
        """
        
        # Get embedding
        try:
            with torch.no_grad():
                embedding = self.model.encode(profile_text, convert_to_tensor=True)
                embedding = self.projection(embedding)
                embedding = F.normalize(embedding, p=2, dim=0)
                
                if self.device:
                    embedding = embedding.to(self.device)
                
                # Cache the result
                self.embedding_cache[cache_key] = embedding
                return embedding
                
        except Exception as e:
            self.logger.error(f"Error generating user embedding: {str(e)}")
            return torch.randn(self.embedding_dim, device=self.device)

    async def _get_base_movie_embedding(self, movie_info: Dict) -> torch.Tensor:
        """Generate base movie embedding with minimal processing."""
        # Create cache key from movie info
        cache_key = f"movie_{movie_info.get('id', '')}_{movie_info.get('title', '')}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Extract movie genres
        genres = movie_info.get('genres', [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split('|')]
        elif not isinstance(genres, list):
            genres = []
            
        # Create concise text representation
        movie_text = f"""
        Title: {movie_info.get('title', 'Unknown Title')}
        Genres: {', '.join(genres)}
        Description: {movie_info.get('description', '')}
        """
        
        # Get embedding
        try:
            with torch.no_grad():
                embedding = self.model.encode(movie_text, convert_to_tensor=True)
                embedding = self.projection(embedding)
                embedding = F.normalize(embedding, p=2, dim=0)
                
                if self.device:
                    embedding = embedding.to(self.device)
                
                # Cache the result
                self.embedding_cache[cache_key] = embedding
                return embedding
                
        except Exception as e:
            self.logger.error(f"Error generating movie embedding: {str(e)}")
            return torch.randn(self.embedding_dim, device=self.device)
            
    async def get_movie_embedding(self, 
                                movie_info: Dict,
                                movie_tags: Optional[pd.DataFrame] = None) -> torch.Tensor:
        """Enhanced movie embedding with tag information."""
        # Get base movie embedding
        base_embedding = await self._get_base_movie_embedding(movie_info)
        
        # Get tag-based embedding if available
        if movie_tags is not None and not movie_tags.empty:
            tag_embedding = await self.process_movie_tags(movie_tags)
            # Combine embeddings with equal weights
            combined = (base_embedding + tag_embedding) / 2
            return F.normalize(combined, p=2, dim=0)
        
        return base_embedding
    
    def clear_cache(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.genre_processor.genre_cache.clear()
        self.genre_processor.similarity_cache.clear()
        self.genre_processor.cache_key = None
        self.logger.info("All caches cleared")