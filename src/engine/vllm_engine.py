"""
vLLM-based inference engine with continuous batching and optimization.
"""
import asyncio
import uuid
from typing import List, Dict, Optional, AsyncIterator
import torch

# Try vLLM import, fallback to transformers if not available
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, using fallback implementation")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class VLLMEngine:
    """High-performance LLM inference engine with vLLM or fallback."""
    
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
        """Initialize inference engine.
        
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
        
        logger.info(f"Initializing inference engine with model: {model_name}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Quantization: {quantization}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if VLLM_AVAILABLE:
            self._init_vllm(gpu_memory_utilization, enable_prefix_caching, **kwargs)
        else:
            self._init_fallback()
        
        self._ready = True
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "batch_count": 0,
            "cache_hits": 0
        }
        
        logger.info("Inference engine initialized successfully")
    
    def _init_vllm(self, gpu_memory_utilization, enable_prefix_caching, **kwargs):
        """Initialize vLLM engine."""
        logger.info("Using vLLM engine")
        
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=self.quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=self.max_batch_size * 512,
            max_num_seqs=self.max_batch_size,
            enable_prefix_caching=enable_prefix_caching,
            **kwargs
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.use_vllm = True
    
    def _init_fallback(self):
        """Initialize fallback transformers engine."""
        logger.info("Using transformers fallback engine")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with appropriate settings
        load_kwargs = {"device_map": "auto"}
        
        if self.quantization == "int8":
            load_kwargs["load_in_8bit"] = True
        elif self.quantization == "int4":
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
        
        self.use_vllm = False
    
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
        """Generate completion for a single prompt."""
        request_id = str(uuid.uuid4())
        
        if self.use_vllm:
            return await self._generate_vllm(
                prompt, request_id, max_tokens, temperature, 
                top_p, n, stop, presence_penalty, frequency_penalty, **kwargs
            )
        else:
            return await self._generate_fallback(
                prompt, request_id, max_tokens, temperature, 
                top_p, n, stop, **kwargs
            )
    
    async def _generate_vllm(
        self, prompt, request_id, max_tokens, temperature,
        top_p, n, stop, presence_penalty, frequency_penalty, **kwargs
    ) -> Dict:
        """Generate using vLLM."""
        from vllm import SamplingParams
        
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
        
        logger.debug(f"Generating completion for request {request_id}")
        
        # Generate with vLLM
        results = self.engine.generate(prompt, sampling_params, request_id)
        
        choices = []
        total_tokens = 0
        
        async for request_output in results:
            for output in request_output.outputs:
                choices.append({
                    "text": output.text,
                    "index": output.index,
                    "finish_reason": output.finish_reason if hasattr(output, 'finish_reason') else "stop",
                    "logprobs": None
                })
                total_tokens += len(output.token_ids)
        
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
    
    async def _generate_fallback(
        self, prompt, request_id, max_tokens, temperature,
        top_p, n, stop, **kwargs
    ) -> Dict:
        """Generate using transformers fallback."""
        loop = asyncio.get_event_loop()
        
        def _sync_generate():
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            return outputs
        
        outputs = await loop.run_in_executor(None, _sync_generate)
        
        choices = []
        for i, output in enumerate(outputs):
            generated_text = output["generated_text"][len(prompt):]
            choices.append({
                "text": generated_text,
                "index": i,
                "finish_reason": "stop",
                "logprobs": None
            })
        
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = sum(len(self.tokenizer.encode(c["text"])) for c in choices)
        
        self._stats["total_requests"] += 1
        self._stats["total_tokens"] += completion_tokens
        
        return {
            "id": request_id,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    
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
