import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals, safe_globals
from tqdm import tqdm

from src.models.base import BaseModel
from src.models.diversity_loss import DiversityAwareLoss
from src.models.neural import HybridNet

from ..configs.model_config import HybridConfig
from ..llm.feature_extractor import LLMFeatureExtractor


class HybridRecommender(BaseModel):
    """Hybrid recommender system combining collaborative filtering with LLM features."""

    def __init__(self, config: HybridConfig):
        """Initialize the hybrid recommender."""
        super().__init__(config)
        self.l2_lambda = config.l2_reg  # L2 regularization strength

        # Initialize neural network with stronger regularization
        self.model = HybridNet(
            n_users=config.n_users,
            n_items=config.n_items,
            n_factors=config.n_factors,
            llm_dim=config.llm_embedding_dim,
            dropout=0.4,
        ).to(self.device)

        # Initialize optimizer with higher learning rate and weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-6  # Number of epochs
        )

        # Initialize feature extractor
        self.feature_extractor = LLMFeatureExtractor(
            model_name=config.llm_model_name,
            embedding_dim=config.llm_embedding_dim,
            device=config.device,
        )

        # Initialize loss function
        self.criterion = DiversityAwareLoss(
            diversity_lambda=self.l2_lambda, rating_lambda=self.l2_lambda
        ).to(self.device)

        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler(self.device)

        # Feature cache
        self.feature_cache = {}

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Training history and state
        self.history = []
        self.steps = 0
        self.accumulation_steps = 4

    def _compute_l2_loss(self):
        """Calculate L2 regularization loss for model parameters."""
        l2_loss = 0.0
        for name, param in self.model.named_parameters():
            if "weight" in name:  # Only apply to weights, not biases
                l2_loss += torch.norm(param, p=2)
        return l2_loss

    async def _get_batch_llm_features(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        movie_info_dict: Dict,
        user_history_dict: Dict,
        user_tags_dict: Optional[Dict] = None,
        movie_tags_dict: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LLM features for a batch of users and items with caching."""
        user_results = []
        item_results = []

        # Process users with caching
        for uid in users:
            user_id = uid.item()
            cache_key = f"user_{user_id}"

            if cache_key in self.feature_cache:
                user_results.append(self.feature_cache[cache_key])
            else:
                user_history = user_history_dict[user_id]
                user_tags = user_tags_dict.get(user_id) if user_tags_dict else None
                embedding = await self.feature_extractor.get_user_preference_embedding(
                    user_history, user_tags
                )
                self.feature_cache[cache_key] = embedding
                user_results.append(embedding)

        # Process items with caching
        for iid in items:
            movie_id = iid.item()
            cache_key = f"movie_{movie_id}"

            if cache_key in self.feature_cache:
                item_results.append(self.feature_cache[cache_key])
            else:
                movie_info = movie_info_dict[movie_id]
                movie_tags = movie_tags_dict.get(movie_id) if movie_tags_dict else None
                embedding = await self.feature_extractor.get_movie_embedding(
                    movie_info, movie_tags
                )
                self.feature_cache[cache_key] = embedding
                item_results.append(embedding)

        return (
            torch.stack(user_results).to(self.device),
            torch.stack(item_results).to(self.device),
        )

    async def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        movie_info_dict: Dict,
        user_history_dict: Dict,
        user_tags_dict: Optional[Dict] = None,
        movie_tags_dict: Optional[Dict] = None,
    ) -> float:
        """Enhanced training step with diversity-aware loss."""
        user_ids = batch["user_id"].to(self.device)
        item_ids = batch["item_id"].to(self.device)
        ratings = batch["rating"].to(self.device)

        user_llm_embeds, item_llm_embeds = await self._get_batch_llm_features(
            user_ids,
            item_ids,
            movie_info_dict,
            user_history_dict,
            user_tags_dict,
            movie_tags_dict,
        )

        with torch.amp.autocast(self.device.type):
            predictions = self.model(
                user_ids, item_ids, user_llm_embeds, item_llm_embeds
            )

            # Calculate loss with diversity consideration
            pred_loss = self.criterion(predictions, ratings, item_ids)

            # Add L2 regularization loss
            l2_loss = self._compute_l2_loss()
            total_loss = pred_loss + self.l2_lambda * l2_loss

            # Scale loss for accumulation
            total_loss = total_loss / self.accumulation_steps

        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()

        if self.steps % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.steps += 1
        return total_loss.item() * self.accumulation_steps

    async def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
        movie_info_dict: Dict,
        user_history_dict: Dict,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad(), torch.amp.autocast(self.device.type):
            progress_bar = tqdm(
                dataloader, desc="Validation batches", leave=False, position=0
            )

            for batch in progress_bar:
                user_ids = batch["user_id"].to(self.device)
                item_ids = batch["item_id"].to(self.device)
                ratings = batch["rating"].to(self.device)

                user_llm_embeds, item_llm_embeds = await self._get_batch_llm_features(
                    user_ids, item_ids, movie_info_dict, user_history_dict
                )

                predictions = self.model(
                    user_ids, item_ids, user_llm_embeds, item_llm_embeds
                )

                # Use only base MSE loss for validation
                loss = F.mse_loss(predictions, ratings)
                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())

                progress_bar.set_description(f"Validation (loss: {loss.item():.4f})")

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

        self.logger.info(f"Validation loss: {total_loss / len(dataloader):.4f}")
        self.logger.info(f"Validation MAE: {mae:.4f}")
        self.logger.info(f"Validation RMSE: {rmse:.4f}")
        self.logger.info(f"Validation samples: {len(all_predictions)}")
        self.logger.info(f"Validation batches: {len(dataloader)}")

        return {
            "val_loss": total_loss / len(dataloader),
            "val_mae": mae,
            "val_rmse": rmse,
        }

    async def fit(
        self, train_data: Dict, valid_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Train the model with epoch-level scheduler step."""
        self.model.train()

        epoch_loss = 0
        running_l2 = 0
        n_batches = len(train_data["dataloader"])

        progress_bar = tqdm(
            enumerate(train_data["dataloader"]),
            total=n_batches,
            desc="Training batches",
            leave=False,
            position=0,
        )

        for _batch_idx, batch in progress_bar:
            loss = await self.train_step(
                batch,
                train_data["movie_info_dict"],
                train_data["user_history_dict"],
                train_data.get("user_tags_dict"),
                train_data.get("movie_tags_dict"),
            )
            epoch_loss += loss

            # Calculate L2 norm for monitoring
            l2_norm = self._compute_l2_loss().item()
            running_l2 += l2_norm

            current_lr = self.scheduler.get_last_lr()[0]

            # Update progress bar with detailed info
            progress_bar.set_description(
                f"Training (loss: {loss:.4f}, L2: {l2_norm:.4f}, lr: {current_lr:.6f})"
            )

        # Step the scheduler once per epoch
        self.scheduler.step()

        metrics = {
            "train_loss": epoch_loss / n_batches,
            "l2_norm": running_l2 / n_batches,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        self.logger.info(f"Epoch: {len(self.history) + 1}")
        self.logger.info(f"Epoch training loss: {metrics['train_loss']:.4f}")
        self.logger.info(f"Average L2 norm: {metrics['l2_norm']:.4f}")
        self.logger.info(f"Learning rate: {metrics['learning_rate']:.6f}")

        # Save model after each epoch at neam checkpoint_epoch.pt
        self.save(os.path.join("models", f"checkpoint_{len(self.history)}.pt"))

        if valid_data:
            self.logger.info("Starting validation...")
            val_metrics = await self.validate(
                valid_data["dataloader"],
                valid_data["movie_info_dict"],
                valid_data["user_history_dict"],
            )
            metrics.update(val_metrics)

        self.history.append(metrics)
        return metrics

    async def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        movie_info_dict: Dict,
        user_history_dict: Dict,
    ) -> torch.Tensor:
        """Generate predictions for user-item pairs."""
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            user_llm_embeds, item_llm_embeds = await self._get_batch_llm_features(
                user_ids, item_ids, movie_info_dict, user_history_dict
            )

            predictions = self.model(
                user_ids.to(self.device),
                item_ids.to(self.device),
                user_llm_embeds,
                item_llm_embeds,
            )

        return predictions

    def save(self, path: str):
        """Save model state and configuration."""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config.__dict__,  # Config objekt helyett dictionary-t mentünk
            "history": self.history,
            "feature_cache": self.feature_cache,
            "steps": self.steps,
        }

        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model state and configuration."""
        # Add HybridConfig to safe globals
        add_safe_globals([HybridConfig])

        try:
            # First try with weights_only=True
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            self.logger.warning(
                "Could not load with weights_only=True, trying with weights_only=False"
            )
            # If that fails, try with weights_only=False
            with safe_globals([HybridConfig]):
                checkpoint = torch.load(  # nosec B614 - Trusted model checkpoint, created by our training pipeline
                    path, map_location=self.device, weights_only=False
                )

        # Restore config from dictionary
        config_dict = checkpoint["config"]
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                setattr(self.config, key, value)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.history = checkpoint.get("history", [])
        self.feature_cache = checkpoint.get("feature_cache", {})
        self.steps = checkpoint.get("steps", 0)

        self.logger.info(f"Model loaded from {path}")

    def precalculate_embeddings(
        self, movie_info_dict: Dict, user_history_dict: Dict
    ) -> None:
        """Pre-calculate all embeddings for faster training and inference."""
        self.logger.info("Pre-calculating embeddings...")

        async def calculate_all():
            # Pre-calculate user embeddings
            for user_id, history in tqdm(
                user_history_dict.items(), desc="Calculating user embeddings"
            ):
                cache_key = f"user_{user_id}"
                if cache_key not in self.feature_cache:
                    embedding = (
                        await self.feature_extractor.get_user_preference_embedding(
                            history
                        )
                    )
                    self.feature_cache[cache_key] = embedding.to(self.device)

            # Pre-calculate movie embeddings
            for movie_id, info in tqdm(
                movie_info_dict.items(), desc="Calculating movie embeddings"
            ):
                cache_key = f"movie_{movie_id}"
                if cache_key not in self.feature_cache:
                    embedding = await self.feature_extractor.get_movie_embedding(info)
                    self.feature_cache[cache_key] = embedding.to(self.device)

        asyncio.run(calculate_all())
        self.logger.info("Embedding pre-calculation completed")

    async def get_top_n_recommendations(
        self,
        user_id: int,
        movie_info_dict: Dict,
        user_history_dict: Dict,
        n: int = 10,
        exclude_watched: bool = True,
        batch_size: int = 1024,
    ) -> List[Tuple[int, float]]:
        """Generate top-N movie recommendations for a user with batched processing."""
        user_history = user_history_dict[user_id]
        watched_movies = set(user_history["movieId"]) if exclude_watched else set()
        candidate_movies = [
            movie_id
            for movie_id in movie_info_dict.keys()
            if movie_id not in watched_movies
        ]

        # Process in batches
        all_predictions = []
        for i in range(0, len(candidate_movies), batch_size):
            batch_movies = candidate_movies[i : i + batch_size]
            user_ids = torch.tensor([user_id] * len(batch_movies)).to(self.device)
            movie_ids = torch.tensor(batch_movies).to(self.device)

            predictions = await self.predict(
                user_ids, movie_ids, movie_info_dict, user_history_dict
            )
            all_predictions.extend(predictions.cpu().numpy())

        # Sort and return top-N
        movie_scores = list(zip(candidate_movies, all_predictions))
        movie_scores.sort(key=lambda x: x[1], reverse=True)

        return movie_scores[:n]

    async def generate_user_explanation(
        self,
        user_id: int,
        movie_id: int,
        movie_info_dict: Dict,
        user_history_dict: Dict,
    ) -> str:
        """Generate an explanation for why a movie was recommended to a user."""
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            # Get cached embeddings if available
            user_embedding = self.feature_cache.get(f"user_{user_id}")
            movie_embedding = self.feature_cache.get(f"movie_{movie_id}")

            if user_embedding is None or movie_embedding is None:
                user_history = user_history_dict[user_id]
                movie_info = movie_info_dict[movie_id]

                if user_embedding is None:
                    user_embedding = (
                        await self.feature_extractor.get_user_preference_embedding(
                            user_history
                        )
                    )
                if movie_embedding is None:
                    movie_embedding = await self.feature_extractor.get_movie_embedding(
                        movie_info
                    )

            # Get prediction
            user_id_tensor = torch.tensor([user_id]).to(self.device)
            movie_id_tensor = torch.tensor([movie_id]).to(self.device)
            user_embedding = user_embedding.unsqueeze(0).to(self.device)
            movie_embedding = movie_embedding.unsqueeze(0).to(self.device)

            # Get model prediction and attention weights
            prediction = self.model(
                user_id_tensor, movie_id_tensor, user_embedding, movie_embedding
            )

        # Generate explanation using LLM
        user_profile = await self.feature_extractor.provider.get_completion(
            self._create_user_profile_prompt(user_history_dict[user_id])
        )

        movie_profile = await self.feature_extractor.provider.get_completion(
            self._create_movie_profile_prompt(movie_info_dict[movie_id])
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

        explanation = await self.feature_extractor.provider.get_completion(
            explanation_prompt
        )
        return explanation
