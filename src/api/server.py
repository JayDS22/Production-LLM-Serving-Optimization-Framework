"""
FastAPI server for LLM inference with streaming, batching, and monitoring.
"""
import asyncio
import time
from typing import List, Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from ..engine.vllm_engine import VLLMEngine
from ..monitoring.metrics import MetricsCollector
from ..utils.config import Config
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

# Global instances
engine: Optional[VLLMEngine] = None
metrics: Optional[MetricsCollector] = None
config: Optional[Config] = None


class CompletionRequest(BaseModel):
    """Request model for text completion."""
    model: str
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=5)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class BatchRequest(BaseModel):
    """Request model for batch processing."""
    prompts: List[str]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class CompletionResponse(BaseModel):
    """Response model for text completion."""
    id: str
    model: str
    choices: List[dict]
    usage: dict
    created: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    uptime_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global engine, metrics, config
    
    # Startup
    logger.info("Starting LLM Serving Framework...")
    config = Config()
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    # Initialize inference engine
    logger.info(f"Loading model: {config.model_name}")
    engine = VLLMEngine(
        model_name=config.model_name,
        tensor_parallel_size=config.tensor_parallel_size,
        quantization=config.quantization_mode,
        max_batch_size=config.max_batch_size,
        gpu_memory_utilization=config.gpu_memory_utilization
    )
    
    logger.info("Server ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
    if engine:
        await engine.shutdown()
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Production LLM Serving Framework",
    description="High-performance LLM inference API with optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    
    return HealthResponse(
        status="healthy" if engine and engine.is_ready() else "unhealthy",
        model_loaded=engine is not None and engine.is_ready(),
        gpu_available=torch.cuda.is_available(),
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
    """Generate text completion."""
    if not engine or not engine.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    start_time = time.time()
    
    try:
        # Record request
        metrics.record_request()
        
        if request.stream:
            return StreamingResponse(
                stream_completion(request),
                media_type="text/event-stream"
            )
        
        # Generate completion
        result = await engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty
        )
        
        # Record metrics
        latency = time.time() - start_time
        metrics.record_latency(latency)
        metrics.record_tokens(result["usage"]["total_tokens"])
        
        return CompletionResponse(
            id=result["id"],
            model=request.model,
            choices=result["choices"],
            usage=result["usage"],
            created=int(time.time())
        )
        
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error generating completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_completion(request: CompletionRequest) -> AsyncIterator[str]:
    """Stream completion tokens."""
    import json
    
    try:
        async for token_data in engine.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        ):
            chunk = {
                "id": token_data["id"],
                "model": request.model,
                "choices": [{
                    "text": token_data["text"],
                    "index": 0,
                    "finish_reason": token_data.get("finish_reason")
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
            if token_data.get("finish_reason"):
                break
                
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        error_chunk = {"error": str(e)}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/batch")
async def batch_completion(request: BatchRequest):
    """Process batch of prompts."""
    if not engine or not engine.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    start_time = time.time()
    
    try:
        metrics.record_batch_request(len(request.prompts))
        
        # Generate completions for all prompts
        results = await engine.generate_batch(
            prompts=request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Record metrics
        latency = time.time() - start_time
        metrics.record_batch_latency(latency, len(request.prompts))
        
        return {
            "results": results,
            "batch_size": len(request.prompts),
            "total_time": latency
        }
        
    except Exception as e:
        metrics.record_error()
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return metrics.export_metrics()


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "model": config.model_name,
        "quantization": config.quantization_mode,
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_batch_size": config.max_batch_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "stats": engine.get_stats(),
        "metrics": metrics.get_summary()
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear KV cache."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    await engine.clear_cache()
    return {"status": "cache cleared"}


if __name__ == "__main__":
    import sys
    
    # Set start time
    app.state.start_time = time.time()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
