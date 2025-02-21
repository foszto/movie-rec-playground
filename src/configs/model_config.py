from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
@dataclass
class ModelConfig:
    """Base configuration for all models."""
    model_type: str
    n_users: int
    n_items: int
    n_factors: int = 100
    learning_rate: float = 1e-4
    dropout: float = 0.2
    weight_decay: float = 0.01
    l2_reg: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cuda"  

    def __post_init__(self):
        """Validáció az inicializálás után."""
        if self.n_users <= 0:
            raise ValueError(f"n_users must be positive, got {self.n_users}")
        if self.n_items <= 0:
            raise ValueError(f"n_items must be positive, got {self.n_items}")
        if self.n_factors <= 0:
            raise ValueError(f"n_factors must be positive, got {self.n_factors}")
        if self.l2_reg < 0:
            raise ValueError(f"l2_reg must be non-negative, got {self.l2_reg}")
        
        # Automatically set device to 'cuda' if available
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"
    
@dataclass
class CollaborativeConfig(ModelConfig):
    """Configuration for collaborative filtering model."""
    dropout: float = 0.2
    
@dataclass
class HybridConfig(ModelConfig):
    """Configuration for hybrid LLM model."""
    # Inherited parameters from ModelConfig
    # - model_type, n_users, n_items, n_factors, learning_rate, dropout, weight_decay, max_grad_norm, device
    
    # LLM specific parameters
    llm_provider: str = "sentence_transformers"
    llm_model_name: str = "all-MiniLM-L6-v2"
    llm_api_key: Optional[str] = None
    llm_embedding_dim: int = 64
    use_cached_embeddings: bool = True
    
    def __post_init__(self):
        """Validáció az inicializálás után."""
        if self.model_type != 'hybrid_llm':
            raise ValueError(f"Invalid model_type for HybridConfig: {self.model_type}")
        
        if self.n_users <= 0:
            raise ValueError(f"Invalid n_users: {self.n_users}")
        
        if self.n_items <= 0:
            raise ValueError(f"Invalid n_items: {self.n_items}")
    
@dataclass
class TrainingConfig:
    """Training configuration."""
    random_seed: int = 42
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    model_save_path: str = "models/saved/"