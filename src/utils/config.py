"""
Configuration management for LLM serving framework.
"""
import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

from .logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class Config:
    """Main configuration class."""
    
    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"
    tensor_parallel_size: int = 1
    quantization_mode: Optional[str] = "int8"
    max_batch_size: int = 256
    gpu_memory_utilization: float = 0.9
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Performance configuration
    enable_flash_attention: bool = True
    enable_kv_cache: bool = True
    kv_cache_dtype: str = "auto"
    max_context_length: int = 4096
    
    # Monitoring configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    
    # Optimization configuration
    batch_timeout_ms: int = 50
    max_waiting_tokens: int = 20
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self._load_from_env()
        self._load_from_file()
        self._validate()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", self.tensor_parallel_size))
        self.quantization_mode = os.getenv("QUANTIZATION_MODE", self.quantization_mode)
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", self.max_batch_size))
        self.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", self.gpu_memory_utilization))
        
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", self.port))
        self.workers = int(os.getenv("WORKERS", self.workers))
        
        self.enable_flash_attention = os.getenv("ENABLE_FLASH_ATTENTION", "true").lower() == "true"
        self.enable_kv_cache = os.getenv("ENABLE_KV_CACHE", "true").lower() == "true"
        
        logger.info(f"Loaded configuration from environment")
    
    def _load_from_file(self, config_path: Optional[Path] = None):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path("config/inference_config.yaml")
        
        if not config_path.exists():
            logger.info(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
            
            # Update configuration
            if "inference" in config_data:
                for key, value in config_data["inference"].items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    def _validate(self):
        """Validate configuration."""
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be between 0.0 and 1.0")
        
        if self.quantization_mode and self.quantization_mode not in ["int8", "int4", "awq", "gptq", None]:
            logger.warning(f"Unknown quantization mode: {self.quantization_mode}")
        
        logger.info("Configuration validated successfully")
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "quantization_mode": self.quantization_mode,
            "max_batch_size": self.max_batch_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "enable_flash_attention": self.enable_flash_attention,
            "enable_kv_cache": self.enable_kv_cache,
            "kv_cache_dtype": self.kv_cache_dtype,
            "max_context_length": self.max_context_length,
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
            "enable_tracing": self.enable_tracing,
            "batch_timeout_ms": self.batch_timeout_ms,
            "max_waiting_tokens": self.max_waiting_tokens
        }
