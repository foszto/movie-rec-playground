import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class HybridNet(nn.Module):
    """Neural network for hybrid recommendation combining collaborative filtering with LLM features."""
    
    def __init__(self, 
                 n_users: int,
                 n_items: int,
                 n_factors: int = 80,
                 llm_dim: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        
        # Collaborative filtering embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Global bias terms
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        
        # Batch normalization
        self.user_bn = nn.BatchNorm1d(n_factors)
        self.item_bn = nn.BatchNorm1d(n_factors)
        
        # LLM feature processing
        self.llm_user_projection = nn.Sequential(
            nn.Linear(llm_dim, n_factors),
            nn.BatchNorm1d(n_factors),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors, n_factors),
            nn.BatchNorm1d(n_factors)
        )
        
        self.llm_item_projection = nn.Sequential(
            nn.Linear(llm_dim, n_factors),
            nn.BatchNorm1d(n_factors),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors, n_factors),
            nn.BatchNorm1d(n_factors)
        )
        
        # Cross-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=n_factors,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion network with increased capacity
        self.fusion = nn.Sequential(
            nn.Linear(n_factors * 4, n_factors * 2),
            nn.BatchNorm1d(n_factors * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors * 2, n_factors),
            nn.BatchNorm1d(n_factors),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_factors.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_factors.weight, mean=0, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        
        for module in [self.llm_user_projection, self.llm_item_projection, self.fusion]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self,
                user_ids: torch.Tensor,
                item_ids: torch.Tensor,
                user_llm_embeds: torch.Tensor,
                item_llm_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hybrid model."""
        # Get collaborative filtering embeddings with batch norm
        user_embed = self.user_bn(self.user_factors(user_ids))
        item_embed = self.item_bn(self.item_factors(item_ids))
        
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
        
        # Element-wise interaction with strong genre emphasis
        cf_interaction = user_embed * item_embed * 1.5
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
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_factors.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_factors.weight, mean=0, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        
        for layer in self.llm_user_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.llm_item_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,
                user_ids: torch.Tensor,
                item_ids: torch.Tensor,
                user_llm_embeds: torch.Tensor,
                item_llm_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hybrid model."""
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