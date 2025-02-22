import torch
import time
from typing import Dict, Any
import logging
from contextlib import contextmanager

class PerformanceProfiler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.timings = {}
        
    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(end - start)
    
    def print_summary(self):
        self.logger.info("\n=== Performance Summary ===")
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            self.logger.info(f"{name}: {avg_time:.4f}s avg ({len(times)} calls)")

def diagnose_model_performance(model, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Detailed performance diagnostics for the model."""
    results = {}
    
    # Check if model is actually on GPU
    results['model_device'] = next(model.parameters()).device
    
    # Check CUDA settings
    results['cuda_settings'] = {
        'cudnn_enabled': torch.backends.cudnn.enabled,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    # Memory usage
    if torch.cuda.is_available():
        results['memory'] = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
        }
    
    # Check batch tensor devices
    results['batch_devices'] = {
        key: tensor.device for key, tensor in batch.items()
    }
    
    return results

def optimize_cuda_settings():
    """Configure optimal CUDA settings."""
    if torch.cuda.is_available():
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        
        # Use TF32 precision if available (on Ampere GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        return True
    return False

# Add this to your HybridRecommender class
async def profile_training_step(self, batch: Dict[str, torch.Tensor], **kwargs):
    """Profile a single training step to identify bottlenecks."""
    profiler = PerformanceProfiler()
    
    with profiler.timer("total_step"):
        # Data transfer to GPU
        with profiler.timer("data_transfer"):
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            ratings = batch['rating'].to(self.device)
        
        # LLM feature extraction
        with profiler.timer("llm_features"):
            user_llm_embeds, item_llm_embeds = await self._get_batch_llm_features(
                user_ids, item_ids, **kwargs
            )
        
        # Forward pass
        with profiler.timer("forward_pass"):
            predictions = self.model(
                user_ids,
                item_ids,
                user_llm_embeds,
                item_llm_embeds
            )
            loss = self.criterion(predictions, ratings)
        
        # Backward pass
        with profiler.timer("backward_pass"):
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
    
    profiler.print_summary()
    return loss.item()