from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text."""
        pass
    
    @abstractmethod
    async def get_completion(self, prompt: str) -> str:
        """Get text completion."""
        pass