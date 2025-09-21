"""
vLLM-based inference engine with continuous batching and optimization.
"""
import asyncio
import uuid
from typing import List, Dict, Optional, AsyncIterator
import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from transformers import AutoTokenizer

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class VLLMEngine:
    """High-performance LLM inference engine using vLLM."""
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_batch_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        **kwargs
    ):
        """Initialize vLLM engine.
        
        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs for tensor parallelism
            quantization: Quantization method (int8, int4, awq, gptq)
            max_batch_size: Maximum batch size for continuous batching
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            enable_prefix_caching: Enable KV-cache prefix caching
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization
        self.max_batch_size = max_batch_size
        
        logger.info(f"Initializing vLLM engine with model: {model_name}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Quantization: {quantization}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure async engine
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_batch_size * 512,
            max_num_seqs=max_batch_size,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=True,
            max_model_len=4096,
            **kwargs
        )
        
        # Create async engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._ready = True
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "batch_count": 0,
            "cache_hits": 0
        }
        
        logger.info("vLLM engine initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if engine is ready."""
        return self._ready
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        **kwargs
    ) -> Dict:
        """Generate completion for a single prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            
        Returns:
            Dict containing generated text and metadata
        """
        request_id = str(uuid.uuid4())
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **kwargs
        )
        
        # Generate
        logger.debug(f"Generating completion for request {request_id}")
        results = await self.engine.generate(prompt, sampling_params, request_id)
        
        # Process results
        choices = []
        total_tokens = 0
        
        async for request_output in results:
            for output in request_output.outputs:
                choices.append({
                    "text": output.text,
                    "index": output.index,
                    "finish_reason": output.finish_reason.value if output.finish_reason else None,
                    "logprobs": output.logprobs
                })
                total_tokens += len(output.token_ids)
        
        # Update stats
        self._stats["total_requests"] += 1
        self._stats["total_tokens"] += total_tokens
        
        return {
            "id": request_id,
            "choices": choices,
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(prompt)),
                "completion_tokens": total_tokens,
                "total_tokens": len(self.tokenizer.encode(prompt)) + total_tokens
            }
        }
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[Dict]:
        """Stream generation tokens in real-time.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            
        Yields:
            Dict containing token data
        """
        request_id = str(uuid.uuid4())
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs
        )
        
        logger.debug(f"Starting streaming generation for request {request_id}")
        
        results = self.engine.generate(prompt, sampling_params, request_id)
        
        async for request_output in results:
            for output in request_output.outputs:
                yield {
                    "id": request_id,
                    "text": output.text,
                    "finish_reason": output.finish_reason.value if output.finish_reason else None
                }
    
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[Dict]:
        """Generate completions for batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of completion results
        """
        logger.info(f"Processing batch of {len(prompts)} prompts")
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Generate all completions
        tasks = []
        for prompt in prompts:
            request_id = str(uuid.uuid4())
            tasks.append(self.engine.generate(prompt, sampling_params, request_id))
        
        # Await all results
        results = []
        for task in tasks:
            async for request_output in task:
                result = {
                    "text": request_output.outputs[0].text,
                    "finish_reason": request_output.outputs[0].finish_reason.value if request_output.outputs[0].finish_reason else None
                }
                results.append(result)
        
        self._stats["batch_count"] += 1
        self._stats["total_requests"] += len(prompts)
        
        return results
    
    async def clear_cache(self):
        """Clear KV cache."""
        logger.info("Clearing KV cache")
        # vLLM manages cache automatically, but we can trigger cleanup
        await self.engine.abort_all()
        self._stats["cache_hits"] = 0
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            **self._stats,
            "model": self.model_name,
            "quantization": self.quantization,
            "tensor_parallel_size": self.tensor_parallel_size
        }
    
    async def shutdown(self):
        """Shutdown engine gracefully."""
        logger.info("Shutting down vLLM engine")
        self._ready = False
        # Cleanup resources
        if hasattr(self, 'engine'):
            await self.engine.abort_all()
        logger.info("vLLM engine shutdown complete")
