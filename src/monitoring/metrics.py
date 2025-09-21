"""
Prometheus metrics collection for LLM serving.
"""
import time
from typing import Dict, List
from collections import defaultdict
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    """Collect and export metrics for monitoring."""
    
    def __init__(self):
        """Initialize metrics collectors."""
        
        # Request metrics
        self.request_counter = Counter(
            'llm_requests_total',
            'Total number of inference requests',
            ['status']
        )
        
        self.batch_request_counter = Counter(
            'llm_batch_requests_total',
            'Total number of batch requests'
        )
        
        # Latency metrics
        self.latency_histogram = Histogram(
            'llm_inference_latency_seconds',
            'Inference latency distribution',
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.batch_latency_histogram = Histogram(
            'llm_batch_latency_seconds',
            'Batch inference latency distribution',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        )
        
        # Throughput metrics
        self.throughput_gauge = Gauge(
            'llm_throughput_tokens_per_second',
            'Token generation throughput'
        )
        
        self.tokens_counter = Counter(
            'llm_tokens_generated_total',
            'Total tokens generated'
        )
        
        # Batch size metrics
        self.batch_size_gauge = Gauge(
            'llm_batch_size',
            'Current batch size'
        )
        
        # GPU metrics
        self.gpu_memory_gauge = Gauge(
            'llm_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id']
        )
        
        self.gpu_utilization_gauge = Gauge(
            'llm_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )
        
        # Active requests
        self.active_requests_gauge = Gauge(
            'llm_active_requests',
            'Number of active requests'
        )
        
        # Error metrics
        self.error_counter = Counter(
            'llm_errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        # Internal tracking
        self._latencies: List[float] = []
        self._throughputs: List[float] = []
        self._batch_sizes: List[int] = []
        self._start_time = time.time()
        self._total_tokens = 0
        self._window_size = 100
        
        logger.info("Metrics collector initialized")
    
    def record_request(self, status: str = "success"):
        """Record a request."""
        self.request_counter.labels(status=status).inc()
        self.active_requests_gauge.inc()
    
    def record_batch_request(self, batch_size: int):
        """Record a batch request."""
        self.batch_request_counter.inc()
        self.batch_size_gauge.set(batch_size)
        self._batch_sizes.append(batch_size)
        
        # Keep only recent data
        if len(self._batch_sizes) > self._window_size:
            self._batch_sizes.pop(0)
    
    def record_latency(self, latency: float):
        """Record inference latency."""
        self.latency_histogram.observe(latency)
        self._latencies.append(latency)
        self.active_requests_gauge.dec()
        
        # Keep only recent data
        if len(self._latencies) > self._window_size:
            self._latencies.pop(0)
    
    def record_batch_latency(self, latency: float, batch_size: int):
        """Record batch inference latency."""
        self.batch_latency_histogram.observe(latency)
        self.record_latency(latency)
        self.record_batch_request(batch_size)
    
    def record_tokens(self, num_tokens: int):
        """Record generated tokens."""
        self.tokens_counter.inc(num_tokens)
        self._total_tokens += num_tokens
        
        # Calculate throughput (tokens/sec)
        elapsed_time = time.time() - self._start_time
        if elapsed_time > 0:
            throughput = self._total_tokens / elapsed_time
            self.throughput_gauge.set(throughput)
            self._throughputs.append(throughput)
            
            if len(self._throughputs) > self._window_size:
                self._throughputs.pop(0)
    
    def record_gpu_metrics(self, gpu_id: int, memory_used: int, utilization: float):
        """Record GPU metrics."""
        self.gpu_memory_gauge.labels(gpu_id=str(gpu_id)).set(memory_used)
        self.gpu_utilization_gauge.labels(gpu_id=str(gpu_id)).set(utilization)
    
    def record_error(self, error_type: str = "unknown"):
        """Record an error."""
        self.error_counter.labels(error_type=error_type).inc()
        self.active_requests_gauge.dec()
    
    def get_summary(self) -> Dict:
        """Get metrics summary."""
        if not self._latencies:
            return {
                "status": "no_data",
                "message": "No metrics collected yet"
            }
        
        return {
            "latency": {
                "p50": np.percentile(self._latencies, 50),
                "p95": np.percentile(self._latencies, 95),
                "p99": np.percentile(self._latencies, 99),
                "mean": np.mean(self._latencies),
                "std": np.std(self._latencies)
            },
            "throughput": {
                "current": self._throughputs[-1] if self._throughputs else 0,
                "mean": np.mean(self._throughputs) if self._throughputs else 0,
                "max": np.max(self._throughputs) if self._throughputs else 0
            },
            "batch_size": {
                "current": self._batch_sizes[-1] if self._batch_sizes else 0,
                "mean": np.mean(self._batch_sizes) if self._batch_sizes else 0,
                "max": np.max(self._batch_sizes) if self._batch_sizes else 0
            },
            "tokens": {
                "total": self._total_tokens,
                "rate": self._total_tokens / (time.time() - self._start_time)
            }
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(REGISTRY).decode('utf-8')
    
    def reset(self):
        """Reset metrics."""
        self._latencies.clear()
        self._throughputs.clear()
        self._batch_sizes.clear()
        self._total_tokens = 0
        self._start_time = time.time()
        logger.info("Metrics reset")


class GPUMonitor:
    """Monitor GPU metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._monitoring = False
    
    async def start_monitoring(self, interval: float = 1.0):
        """Start GPU monitoring loop."""
        import torch
        import asyncio
        
        self._monitoring = True
        logger.info("Starting GPU monitoring")
        
        while self._monitoring:
            try:
                for i in range(torch.cuda.device_count()):
                    # Get GPU memory
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    # Get GPU utilization (simplified)
                    utilization = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                    
                    # Record metrics
                    self.metrics.record_gpu_metrics(
                        gpu_id=i,
                        memory_used=memory_allocated,
                        utilization=utilization
                    )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring GPU: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self._monitoring = False
        logger.info("Stopping GPU monitoring")
