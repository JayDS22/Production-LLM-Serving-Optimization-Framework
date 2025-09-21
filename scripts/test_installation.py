#!/usr/bin/env python3
"""
Test installation and verify all components work.
"""
import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - FAILED: {e}")
        return False

def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA - Available (Devices: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
            return True
        else:
            print("! CUDA - Not available (CPU mode)")
            return False
    except Exception as e:
        print(f"✗ CUDA - Error: {e}")
        return False

def test_model_loading():
    """Test loading a small model."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("\nTesting model loading (GPT-2)...")
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Test inference
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=20)
        result = tokenizer.decode(outputs[0])
        
        print(f"✓ Model loading and inference - OK")
        print(f"  Sample output: {result[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Model loading - FAILED: {e}")
        return False

def test_vllm():
    """Test vLLM availability."""
    try:
        import vllm
        print(f"✓ vLLM - Available (version {vllm.__version__})")
        return True
    except ImportError:
        print("! vLLM - Not installed (will use fallback)")
        return False

def test_api_imports():
    """Test API component imports."""
    print("\nTesting API components...")
    try:
        from fastapi import FastAPI
        from uvicorn import Config
        from pydantic import BaseModel
        print("✓ FastAPI components - OK")
        return True
    except Exception as e:
        print(f"✗ FastAPI components - FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("LLM Serving Framework - Installation Test")
    print("="*50)
    
    print("\n[Core Dependencies]")
    results = []
    
    results.append(test_import("torch", "PyTorch"))
    results.append(test_import("transformers", "Transformers"))
    results.append(test_import("fastapi", "FastAPI"))
    results.append(test_import("prometheus_client", "Prometheus"))
    results.append(test_import("loguru", "Loguru"))
    
    print("\n[Optional Dependencies]")
    test_vllm()
    test_import("bitsandbytes", "BitsAndBytes (quantization)")
    
    print("\n[Hardware]")
    cuda_available = test_cuda()
    
    print("\n[Functionality Tests]")
    results.append(test_api_imports())
    
    # Only test model loading if user confirms (downloads data)
    import os
    if os.getenv("TEST_MODEL_LOADING", "").lower() == "true":
        results.append(test_model_loading())
    else:
        print("! Model loading test skipped (set TEST_MODEL_LOADING=true to enable)")
    
    print("\n" + "="*50)
    if all(results):
        print("✓ All critical tests passed!")
        print("\nYou can now:")
        print("1. Start the server: python -m uvicorn src.api.server:app --reload")
        print("2. Run tests: pytest tests/")
        print("3. Use Docker: docker-compose up")
        return 0
    else:
        print("✗ Some tests failed - check output above")
        print("\nTo fix:")
        print("1. Ensure all dependencies are installed: pip install -r requirements-core.txt")
        print("2. For CUDA support, install PyTorch with CUDA")
        print("3. For vLLM, run: pip install vllm")
        return 1

if __name__ == "__main__":
    sys.exit(main())
