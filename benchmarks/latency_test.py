"""
Latency benchmarking for LLM inference.
"""
import asyncio
import time
import argparse
import numpy as np
from typing import List, Dict
import aiohttp
import json
from datetime import datetime

class LatencyBenchmark:
    """Benchmark inference latency."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-2-7b-hf"
    ):
        self.base_url = base_url
        self.model = model
        self.latencies: List[float] = []
    
    async def single_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int = 256
    ) -> Dict:
        """Send single inference request and measure latency."""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
            ) as response:
                result = await response.json()
                latency = time.time() - start_time
                
                return {
                    "success": True,
                    "latency": latency,
                    "tokens": result.get("usage", {}).get("completion_tokens", 0)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    async def run_concurrent_requests(
        self,
        num_requests: int,
        concurrent_users: int,
        prompt: str = "Explain quantum computing in simple terms:"
    ):
        """Run concurrent requests and collect latency metrics."""
        print(f"Running {num_requests} requests with {concurrent_users} concurrent users...")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(num_requests):
                task = self.single_request(session, prompt)
                tasks.append(task)
                
                # Control concurrency
                if len(tasks) >= concurrent_users:
                    results = await asyncio.gather(*tasks)
                    self._process_results(results)
                    tasks = []
                
                # Small delay to simulate realistic load
                await asyncio.sleep(0.01)
            
            # Process remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks)
                self._process_results(results)
    
    def _process_results(self, results: List[Dict]):
        """Process and store results."""
        for result in results:
            if result["success"]:
                self.latencies.append(result["latency"])
    
    def print_report(self):
        """Print benchmark report."""
        if not self.latencies:
            print("No successful requests completed")
            return
        
        latencies_ms = [l * 1000 for l in self.latencies]
        
        print("\n" + "="*60)
        print("LATENCY BENCHMARK RESULTS")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Total Requests: {len(self.latencies)}")
        print(f"Successful Requests: {len(self.latencies)}")
        print(f"\nLatency Distribution (ms):")
        print(f"  Mean:   {np.mean(latencies_ms):.2f}")
        print(f"  Median: {np.median(latencies_ms):.2f}")
        print(f"  P50:    {np.percentile(latencies_ms, 50):.2f}")
        print(f"  P95:    {np.percentile(latencies_ms, 95):.2f}")
        print(f"  P99:    {np.percentile(latencies_ms, 99):.2f}")
        print(f"  P99.9:  {np.percentile(latencies_ms, 99.9):.2f}")
        print(f"  Min:    {np.min(latencies_ms):.2f}")
        print(f"  Max:    {np.max(latencies_ms):.2f}")
        print(f"  Std:    {np.std(latencies_ms):.2f}")
        
        # Throughput
        throughput = len(self.latencies) / sum(self.latencies)
        print(f"\nThroughput: {throughput:.2f} requests/sec")
        print("="*60 + "\n")
    
    def export_results(self, filename: str = "latency_results.json"):
        """Export results to JSON."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "total_requests": len(self.latencies),
            "latency_ms": {
                "mean": float(np.mean([l * 1000 for l in self.latencies])),
                "median": float(np.median([l * 1000 for l in self.latencies])),
                "p50": float(np.percentile([l * 1000 for l in self.latencies], 50)),
                "p95": float(np.percentile([l * 1000 for l in self.latencies], 95)),
                "p99": float(np.percentile([l * 1000 for l in self.latencies], 99)),
                "min": float(np.min([l * 1000 for l in self.latencies])),
                "max": float(np.max([l * 1000 for l in self.latencies])),
                "std": float(np.std([l * 1000 for l in self.latencies]))
            },
            "raw_latencies": self.latencies
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filename}")


async def main():
    parser = argparse.ArgumentParser(description="LLM Latency Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--num-requests", type=int, default=1000, help="Number of requests")
    parser.add_argument("--concurrent-users", type=int, default=100, help="Concurrent users")
    parser.add_argument("--export", default="latency_results.json", help="Export filename")
    
    args = parser.parse_args()
    
    benchmark = LatencyBenchmark(base_url=args.url, model=args.model)
    
    await benchmark.run_concurrent_requests(
        num_requests=args.num_requests,
        concurrent_users=args.concurrent_users
    )
    
    benchmark.print_report()
    benchmark.export_results(args.export)


if __name__ == "__main__":
    asyncio.run(main())
