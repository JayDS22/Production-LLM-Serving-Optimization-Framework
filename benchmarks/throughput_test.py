"""
Throughput benchmarking for LLM inference.
"""
import asyncio
import time
import argparse
from typing import List, Dict
import aiohttp
from datetime import datetime
import json

class ThroughputBenchmark:
    """Benchmark token generation throughput."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-2-7b-hf"
    ):
        self.base_url = base_url
        self.model = model
        self.total_tokens = 0
        self.total_requests = 0
        self.start_time = None
        self.end_time = None
    
    async def sustained_load(
        self,
        duration: int = 300,
        ramp_up: int = 60,
        target_rps: int = 100,
        prompt: str = "Write a detailed explanation of machine learning:"
    ):
        """Run sustained load test."""
        print(f"Starting throughput test for {duration}s with {ramp_up}s ramp-up...")
        print(f"Target: {target_rps} requests/sec")
        
        self.start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            elapsed = 0
            while elapsed < duration:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Calculate current RPS based on ramp-up
                if elapsed < ramp_up:
                    current_rps = int(target_rps * (elapsed / ramp_up))
                else:
                    current_rps = target_rps
                
                # Send requests
                for _ in range(max(1, current_rps // 10)):
                    task = asyncio.create_task(
                        self.send_request(session, prompt)
                    )
                    tasks.append(task)
                
                # Clean up completed tasks
                tasks = [t for t in tasks if not t.done()]
                
                await asyncio.sleep(0.1)
                
                # Print progress
                if int(elapsed) % 10 == 0:
                    print(f"Progress: {int(elapsed)}s / {duration}s - "
                          f"Active requests: {len(tasks)}")
            
            # Wait for remaining tasks
            print("Waiting for remaining requests...")
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = time.time()
    
    async def send_request(self, session: aiohttp.ClientSession, prompt: str):
        """Send single request and track tokens."""
        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.7
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    tokens = result.get("usage", {}).get("total_tokens", 0)
                    self.total_tokens += tokens
                    self.total_requests += 1
                else:
                    print(f"Error: HTTP {response.status}")
        except asyncio.TimeoutError:
            print("Request timeout")
        except Exception as e:
            print(f"Error: {e}")
    
    def print_report(self):
        """Print throughput report."""
        if not self.start_time or not self.end_time:
            print("No benchmark data available")
            return
        
        duration = self.end_time - self.start_time
        
        print("\n" + "="*60)
        print("THROUGHPUT BENCHMARK RESULTS")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Duration: {duration:.2f}s")
        print(f"\nRequests:")
        print(f"  Total: {self.total_requests}")
        print(f"  Per Second: {self.total_requests / duration:.2f}")
        print(f"\nTokens:")
        print(f"  Total: {self.total_tokens}")
        print(f"  Per Second: {self.total_tokens / duration:.2f}")
        print(f"  Per Request: {self.total_tokens / self.total_requests:.2f}")
        print("="*60 + "\n")
    
    def export_results(self, filename: str = "throughput_results.json"):
        """Export results to JSON."""
        duration = self.end_time - self.start_time
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "duration_seconds": duration,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "requests_per_second": self.total_requests / duration,
            "tokens_per_second": self.total_tokens / duration,
            "tokens_per_request": self.total_tokens / self.total_requests if self.total_requests > 0 else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filename}")


async def main():
    parser = argparse.ArgumentParser(description="LLM Throughput Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--duration", type=int, default=300, help="Test duration (seconds)")
    parser.add_argument("--ramp-up", type=int, default=60, help="Ramp-up period (seconds)")
    parser.add_argument("--target-rps", type=int, default=100, help="Target requests/sec")
    parser.add_argument("--export", default="throughput_results.json", help="Export filename")
    
    args = parser.parse_args()
    
    benchmark = ThroughputBenchmark(base_url=args.url, model=args.model)
    
    await benchmark.sustained_load(
        duration=args.duration,
        ramp_up=args.ramp_up,
        target_rps=args.target_rps
    )
    
    benchmark.print_report()
    benchmark.export_results(args.export)


if __name__ == "__main__":
    asyncio.run(main())
