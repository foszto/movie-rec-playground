from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from src.configs.model_config import ModelConfig

class BaseModel(ABC):
    """Abstract base class for all recommender models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_device(config)

        self.feature_cache = {}
        self.accumulation_steps = 1

    def _init_device(self, config: ModelConfig):
        """Initialize device."""

        if config.device is not None:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cuda':
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    
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

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization settings and status."""
        status = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'mixed_precision_enabled': True,
            'gradient_accumulation_steps': self.accumulation_steps,
            'feature_cache_size': len(self.feature_cache),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'tf32_enabled': torch.backends.cuda.matmul.allow_tf32
        }
        
        if torch.cuda.is_available():
            status.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**2
            })
        
        return status