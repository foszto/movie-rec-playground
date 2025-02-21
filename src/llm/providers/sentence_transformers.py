from ..base import BaseLLMProvider
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

class SentenceTransformerProvider(BaseLLMProvider):
    """Use local Sentence Transformers for embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    async def get_embedding(self, text: str) -> torch.Tensor:
        embedding = self.model.encode(text, convert_to_tensor=True)
        return F.normalize(embedding, p=2, dim=0)
    
    async def get_completion(self, prompt: str) -> str:
        raise NotImplementedError("This provider only supports embeddings")