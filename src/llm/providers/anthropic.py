from ..base import BaseLLMProvider
import torch
import torch.nn.functional as F
from anthropic import Anthropic
import logging

class AnthropicProvider(BaseLLMProvider):
    """Use Anthropic Claude for completions and embeddings."""
    
    def __init__(self, 
                 api_key: str,
                 model_name: str = "claude-3-haiku-20240307"):
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.embedding_dim = 64
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_embedding(self, text: str) -> torch.Tensor:
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": text
                }]
            )
            # Note: This is a placeholder until Anthropic provides official embedding API
            embedding = torch.randn(self.embedding_dim)
            return F.normalize(embedding, p=2, dim=0)
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            raise
    
    async def get_completion(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Error getting completion: {str(e)}")
            raise