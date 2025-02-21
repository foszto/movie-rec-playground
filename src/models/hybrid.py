#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging
import asyncio
from pathlib import Path
import numpy as np
from ..llm.feature_extractor import LLMFeatureExtractor
from ..configs.model_config import HybridConfig

class HybridNet(nn.Module):
    """Neural network for hybrid recommendation combining collaborative filtering with LLM features."""
    
    def __init__(self, 
                 n_users: int,
                 n_items: int,
                 n_factors: int = 80,
                 llm_dim: int = 128,
                 dropout: float = 0.4):
        """
        Initialize the hybrid neural network.
        
        Args:
            n_users: Number of users in the dataset
            n_items: Number of items (movies) in the dataset
            n_factors: Dimensionality of embedding vectors
            llm_dim: Dimension of LLM embeddings
            dropout: Dropout rate
        """
        super().__init__()
        
        # Collaborative filtering embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Global bias terms
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        
        # LLM feature processing
        self.llm_user_projection = nn.Sequential(
            nn.Linear(llm_dim, n_factors),
            nn.LayerNorm(n_factors),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.llm_item_projection = nn.Sequential(
            nn.Linear(llm_dim, n_factors),
            nn.LayerNorm(n_factors),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=n_factors,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(n_factors * 4, n_factors * 2),  # 4 = CF + LLM + Attention + Element-wise
            nn.LayerNorm(n_factors * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors * 2, n_factors),
            nn.LayerNorm(n_factors),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.user_factors.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_factors.weight, mean=0, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        
        # Initialize projection layers
        for layer in self.llm_user_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.llm_item_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize fusion layers
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,
                user_ids: torch.Tensor,
                item_ids: torch.Tensor,
                user_llm_embeds: torch.Tensor,
                item_llm_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            user_llm_embeds: LLM embeddings for users
            item_llm_embeds: LLM embeddings for items
            
        Returns:
            Tensor of predicted ratings
        """
        # Get collaborative filtering embeddings
        user_embed = self.user_factors(user_ids)
        item_embed = self.item_factors(item_ids)
        
        # Get bias terms
        user_bias = self.user_biases(user_ids).squeeze(-1)
        item_bias = self.item_biases(item_ids).squeeze(-1)
        
        # Project LLM embeddings
        user_llm_proj = self.llm_user_projection(user_llm_embeds)
        item_llm_proj = self.llm_item_projection(item_llm_embeds)
        
        # Compute attention between user and item embeddings
        user_context = torch.stack([user_embed, user_llm_proj], dim=1)
        item_context = torch.stack([item_embed, item_llm_proj], dim=1)
        
        attended_user, _ = self.attention(
            user_context,
            item_context,
            item_context
        )
        
        attended_item, _ = self.attention(
            item_context,
            user_context,
            user_context
        )
        
        # Element-wise interaction
        cf_interaction = user_embed * item_embed
        llm_interaction = user_llm_proj * item_llm_proj
        
        # Combine all features
        combined = torch.cat([
            cf_interaction,
            llm_interaction,
            attended_user.mean(dim=1),
            attended_item.mean(dim=1)
        ], dim=-1)
        
        # Final prediction
        prediction = (
            self.fusion(combined).squeeze(-1) +
            user_bias +
            item_bias +
            self.global_bias
        )
        
        return prediction

class HybridRecommender:
    """Hybrid recommender system combining collaborative filtering with LLM features."""
    
    def __init__(self, config: HybridConfig):
        """
        Initialize the hybrid recommender.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize neural network (csak egyszer!)
        self.model = HybridNet(
            n_users=config.n_users,
            n_items=config.n_items,
            n_factors=config.n_factors,
            llm_dim=config.llm_embedding_dim,
            dropout=config.dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize LLM feature extractor
        self.feature_extractor = LLMFeatureExtractor(
            model_name=config.llm_model_name,
            embedding_dim=config.llm_embedding_dim,
            device=config.device
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Training history
        self.history = []
    
    async def _get_batch_llm_features(self,
                                    users: torch.Tensor,
                                    items: torch.Tensor,
                                    movie_info_dict: Dict,
                                    user_history_dict: Dict
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LLM features for a batch of users and items."""
        # Create tasks for all embeddings
        user_tasks = [
            self.feature_extractor.get_user_preference_embedding(
                user_history_dict[uid.item()]
            )
            for uid in users
        ]
        
        item_tasks = [
            self.feature_extractor.get_movie_embedding(
                movie_info_dict[iid.item()]
            )
            for iid in items
        ]
        
        # Run all tasks concurrently
        user_results = await asyncio.gather(*user_tasks)
        item_results = await asyncio.gather(*item_tasks)
        
        return (
            torch.stack(user_results).to(self.device),
            torch.stack(item_results).to(self.device)
        )
    
    async def train_step(self,
                        batch: Dict[str, torch.Tensor],
                        movie_info_dict: Dict,
                        user_history_dict: Dict) -> float:
        """Perform a single training step."""
        self.model.train()
        
        # Move batch to device
        user_ids = batch['user_id'].to(self.device)
        item_ids = batch['item_id'].to(self.device)
        ratings = batch['rating'].to(self.device)
        
        # Get LLM features
        user_llm_embeds, item_llm_embeds = await self._get_batch_llm_features(
            user_ids,
            item_ids,
            movie_info_dict,
            user_history_dict
        )
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(
            user_ids,
            item_ids,
            user_llm_embeds,
            item_llm_embeds
        )
        
        # Compute loss
        loss = self.criterion(predictions, ratings)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    async def validate(self,
                    dataloader: torch.utils.data.DataLoader,
                    movie_info_dict: Dict,
                    user_history_dict: Dict) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, 
                            desc="Validation batches",
                            leave=False,
                            position=0)
            
            for batch in progress_bar:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                user_llm_embeds, item_llm_embeds = await self._get_batch_llm_features(
                    user_ids,
                    item_ids,
                    movie_info_dict,
                    user_history_dict
                )
                
                predictions = self.model(
                    user_ids,
                    item_ids,
                    user_llm_embeds,
                    item_llm_embeds
                )
                
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
                
                # Frissítsük a progress bar leírását
                progress_bar.set_description(f"Validation (loss: {loss.item():.4f})")
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        
        return {
            'val_loss': total_loss / len(dataloader),
            'val_mae': mae,
            'val_rmse': rmse
        }
    
    async def fit(self,
                train_data: Dict,
                valid_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            train_data: Dictionary containing training data
            valid_data: Optional dictionary containing validation data
        
        Returns:
            Dictionary of training metrics
        """
        epoch_loss = 0
        n_batches = len(train_data['dataloader'])
        
        # Train loop
        progress_bar = tqdm(enumerate(train_data['dataloader']), 
                        total=n_batches,
                        desc="Training batches",
                        leave=False,
                        position=0)
        
        for batch_idx, batch in progress_bar:
            loss = await self.train_step(
                batch,
                train_data['movie_info_dict'],
                train_data['user_history_dict']
            )
            epoch_loss += loss
            
            # Update progress bar
            progress_bar.set_description(f"Training (loss: {loss:.4f})")
        
        metrics = {'train_loss': epoch_loss / n_batches}
        
        if valid_data:
            self.logger.info("Starting validation...")
            val_metrics = await self.validate(
                valid_data['dataloader'],
                valid_data['movie_info_dict'],
                valid_data['user_history_dict']
            )
            metrics.update(val_metrics)
        
        self.history.append(metrics)
        return metrics
    
    async def predict(self,
                     user_ids: torch.Tensor,
                     item_ids: torch.Tensor,
                     movie_info_dict: Dict,
                     user_history_dict: Dict) -> torch.Tensor:
        """Generate predictions for user-item pairs."""
        self.model.eval()
        with torch.no_grad():
            user_llm_embeds, item_llm_embeds = await self._get_batch_llm_features(
                user_ids,
                item_ids,
                movie_info_dict,
                user_history_dict
            )
            
            predictions = self.model(
                user_ids.to(self.device),
                item_ids.to(self.device),
                user_llm_embeds,
                item_llm_embeds
            )
        
        return predictions
    
    def save(self, path: str):
        """Save model state and configuration."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'embedding_cache': self.feature_extractor.embedding_cache
        }
        
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model state and configuration."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.feature_extractor.embedding_cache = checkpoint['embedding_cache']
        
        self.logger.info(f"Model loaded from {path}")

    def calculate_user_embedding_cache(self, user_history_dict: Dict) -> None:
        """Pre-calculate user embeddings for faster inference."""
        self.logger.info("Pre-calculating user embeddings...")
        for user_id, history in user_history_dict.items():
            if user_id not in self.feature_extractor.embedding_cache:
                embedding = asyncio.run(
                    self.feature_extractor.get_user_preference_embedding(history)
                )
                self.feature_extractor.embedding_cache[f"user_{user_id}"] = embedding

    def calculate_movie_embedding_cache(self, movie_info_dict: Dict) -> None:
        """Pre-calculate movie embeddings for faster inference."""
        self.logger.info("Pre-calculating movie embeddings...")
        for movie_id, info in movie_info_dict.items():
            if movie_id not in self.feature_extractor.embedding_cache:
                embedding = asyncio.run(
                    self.feature_extractor.get_movie_embedding(info)
                )
                self.feature_extractor.embedding_cache[f"movie_{movie_id}"] = embedding

    async def get_top_n_recommendations(self,
                                      user_id: int,
                                      movie_info_dict: Dict,
                                      user_history_dict: Dict,
                                      n: int = 10,
                                      exclude_watched: bool = True) -> List[Tuple[int, float]]:
        """
        Generate top-N movie recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            movie_info_dict: Dictionary of movie information
            user_history_dict: Dictionary of user histories
            n: Number of recommendations to generate
            exclude_watched: Whether to exclude already watched movies
            
        Returns:
            List of (movie_id, score) tuples
        """
        user_history = user_history_dict[user_id]
        watched_movies = set(user_history['movieId']) if exclude_watched else set()
        candidate_movies = [
            movie_id for movie_id in movie_info_dict.keys()
            if movie_id not in watched_movies
        ]
        
        # Create batch of user-movie pairs
        user_ids = torch.tensor([user_id] * len(candidate_movies))
        movie_ids = torch.tensor(candidate_movies)
        
        # Get predictions
        predictions = await self.predict(
            user_ids,
            movie_ids,
            movie_info_dict,
            user_history_dict
        )
        
        # Sort movies by predicted rating
        movie_scores = list(zip(candidate_movies, predictions.cpu().numpy()))
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return movie_scores[:n]

    async def generate_user_explanation(self,
                                     user_id: int,
                                     movie_id: int,
                                     movie_info_dict: Dict,
                                     user_history_dict: Dict) -> str:
        """
        Generate an explanation for why a movie was recommended to a user.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            movie_info_dict: Dictionary of movie information
            user_history_dict: Dictionary of user histories
            
        Returns:
            String explanation of the recommendation
        """
        # Get user and movie embeddings
        user_history = user_history_dict[user_id]
        movie_info = movie_info_dict[movie_id]
        
        user_embedding = await self.feature_extractor.get_user_preference_embedding(user_history)
        movie_embedding = await self.feature_extractor.get_movie_embedding(movie_info)
        
        # Get prediction and attention weights
        self.model.eval()
        with torch.no_grad():
            user_id_tensor = torch.tensor([user_id]).to(self.device)
            movie_id_tensor = torch.tensor([movie_id]).to(self.device)
            user_embedding = user_embedding.unsqueeze(0).to(self.device)
            movie_embedding = movie_embedding.unsqueeze(0).to(self.device)
            
            # Get model prediction and attention weights
            prediction = self.model(
                user_id_tensor,
                movie_id_tensor,
                user_embedding,
                movie_embedding
            )
        
        # Generate explanation using LLM
        user_profile = await self.feature_extractor.provider.get_completion(
            self._create_user_profile_prompt(user_history)
        )
        
        movie_profile = await self.feature_extractor.provider.get_completion(
            self._create_movie_profile_prompt(movie_info)
        )
        
        explanation_prompt = f"""
        Based on the following information, explain why this movie might be a good recommendation:
        
        User Profile:
        {user_profile}
        
        Movie Information:
        {movie_profile}
        
        Predicted Rating: {prediction.item():.2f} out of 5.0
        
        Please provide a concise, natural explanation focusing on:
        1. Key matching elements between user preferences and movie characteristics
        2. Similar movies the user has enjoyed
        3. Unique aspects of the movie that might appeal to this user
        """
        
        explanation = await self.feature_extractor.provider.get_completion(explanation_prompt)
        return explanation

    def _create_user_profile_prompt(self, user_history: pd.DataFrame) -> str:
        """Create a prompt for generating user profile."""
        top_movies = user_history[user_history['rating'] >= 4.0]['title'].tolist()[:5]
        favorite_genres = set()
        for genres in user_history[user_history['rating'] >= 4.0]['genres']:
            favorite_genres.update(genres)
        
        return f"""
        Create a viewer profile based on the following information:
        - Favorite movies: {', '.join(top_movies)}
        - Preferred genres: {', '.join(favorite_genres)}
        - Total movies rated: {len(user_history)}
        - Average rating: {user_history['rating'].mean():.2f}
        """

    def _create_movie_profile_prompt(self, movie_info: Dict) -> str:
        """Create a prompt for generating movie profile."""
        return f"""
        Create a movie profile based on the following information:
        - Title: {movie_info['title']}
        - Genres: {', '.join(movie_info['genres'])}
        - Description: {movie_info.get('description', '')}
        """

    def get_evaluation_metrics(self) -> Dict[str, List[float]]:
        """Get training and validation metrics history."""
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': []
        }
        
        for epoch_metrics in self.history:
            for metric in metrics:
                if metric in epoch_metrics:
                    metrics[metric].append(epoch_metrics[metric])
        
        return metrics

    def get_model_summary(self) -> str:
        """Get a summary of the model architecture and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = [
            "Hybrid Recommender Model Summary:",
            f"- Number of users: {self.config.n_users}",
            f"- Number of items: {self.config.n_items}",
            f"- Embedding dimension: {self.config.n_factors}",
            f"- LLM embedding dimension: {self.config.llm_embedding_dim}",
            f"- Total parameters: {total_params:,}",
            f"- Trainable parameters: {trainable_params:,}",
            f"- Device: {self.device}",
            f"- Training epochs: {len(self.history)}",
            "\nModel Architecture:",
            str(self.model)
        ]
        
        return "\n".join(summary)