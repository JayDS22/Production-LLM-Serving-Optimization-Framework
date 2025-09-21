"""
Test client for LLM serving API.
"""
import asyncio
import aiohttp
import json
from typing import Optional

class LLMClient:
    """Client for testing LLM serving API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self):
        """Check server health."""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def generate(self, prompt: str, max_tokens: int = 256, stream: bool = False):
        """Generate completion."""
        payload = {
            "model": "meta-llama/Llama-2-7b-hf",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": stream
        }
        
        if stream:
            return await self._stream_generate(payload)
        else:
            async with self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                return await response.json()
    
    async def _stream_generate(self, payload: dict):
        """Handle streaming response."""
        async with self.session.post(
            f"{self.base_url}/v1/completions",
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        yield json.loads(data)
    
    async def batch_generate(self, prompts: list, max_tokens: int = 256):
        """Batch generation."""
        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/batch",
            json=payload
        ) as response:
            return await response.json()
    
    async def get_stats(self):
        """Get server statistics."""
        async with self.session.get(f"{self.base_url}/stats") as response:
            return await response.json()


async def test_basic_completion():
    """Test basic completion."""
    print("Testing basic completion...")
    
    async with LLMClient() as client:
        # Health check
        health = await client.health_check()
        print(f"Health: {health}")
        
        # Generate completion
        result = await client.generate(
            "Explain quantum computing in simple terms:",
            max_tokens=100
        )
        
        print(f"\nPrompt: Explain quantum computing in simple terms:")
        print(f"Response: {result['choices'][0]['text']}")
        print(f"Tokens: {result['usage']['total_tokens']}")


async def test_streaming():
    """Test streaming completion."""
    print("\n" + "="*60)
    print("Testing streaming completion...")
    
    async with LLMClient() as client:
        print("Prompt: Write a short story about AI:")
        print("Response: ", end="", flush=True)
        
        async for chunk in await client.generate(
            "Write a short story about AI:",
            max_tokens=200,
            stream=True
        ):
            if chunk.get('choices'):
                text = chunk['choices'][0].get('text', '')
                print(text, end="", flush=True)
        
        print("\n")


async def test_batch_processing():
    """Test batch processing."""
    print("="*60)
    print("Testing batch processing...")
    
    prompts = [
        "Translate to French: Hello, how are you?",
        "Translate to Spanish: Hello, how are you?",
        "Translate to German: Hello, how are you?"
    ]
    
    async with LLMClient() as client:
        result = await client.batch_generate(prompts, max_tokens=50)
        
        print(f"\nBatch size: {result['batch_size']}")
        print(f"Total time: {result['total_time']:.3f}s")
        
        for i, res in enumerate(result['results']):
            print(f"\n[{i+1}] {prompts[i]}")
            print(f"    → {res['text']}")


async def test_performance():
    """Test performance metrics."""
    print("="*60)
    print("Testing performance metrics...")
    
    async with LLMClient() as client:
        # Run multiple requests
        tasks = []
        for i in range(10):
            task = client.generate(f"Count to {i+1}:", max_tokens=50)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Get stats
        stats = await client.get_stats()
        
        print(f"\nServer Statistics:")
        print(f"Model: {stats['model']}")
        print(f"Quantization: {stats['quantization']}")
        print(f"Tensor Parallel Size: {stats['tensor_parallel_size']}")
        
        if 'metrics' in stats:
            metrics = stats['metrics']
            if 'latency' in metrics:
                print(f"\nLatency Metrics:")
                print(f"  P50: {metrics['latency']['p50']*1000:.2f}ms")
                print(f"  P95: {metrics['latency']['p95']*1000:.2f}ms")
                print(f"  P99: {metrics['latency']['p99']*1000:.2f}ms")
            
            if 'throughput' in metrics:
                print(f"\nThroughput:")
                print(f"  Current: {metrics['throughput']['current']:.2f} tokens/s")
                print(f"  Mean: {metrics['throughput']['mean']:.2f} tokens/s")


async def main():
    """Run all tests."""
    print("="*60)
    print("LLM Serving Framework - Test Suite")
    print("="*60)
    
    try:
        await test_basic_completion()
        await test_streaming()
        await test_batch_processing()
        await test_performance()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the server is running on http://localhost:8000")


if __name__ == "__main__":
    asyncio.run(main())
