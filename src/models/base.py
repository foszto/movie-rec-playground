from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

class BaseModel(ABC):
    """Abstract base class for all recommender models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fit(self, train_data: Any, valid_data: Optional[Any] = None) -> Dict[str, float]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Generate predictions."""
        pass
    
    def save(self, path: str):
        """Save model state."""
        self.logger.info(f"Saving model to {path}")
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model state."""
        self.logger.info(f"Loading model from {path}")
        self.load_state_dict(torch.load(path))