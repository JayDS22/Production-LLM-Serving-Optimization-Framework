"""
Unit tests for inference engine.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.eos_token_id = 0
    return tokenizer

@pytest.fixture
def mock_model():
    """Mock model."""
    model = Mock()
    return model

class TestVLLMEngine:
    """Test VLLMEngine class."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization."""
        from src.engine.vllm_engine import VLLMEngine
        
        with patch('src.engine.vllm_engine.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = Mock()
            
            # Test with fallback (no vLLM)
            with patch('src.engine.vllm_engine.VLLM_AVAILABLE', False):
                with patch('src.engine.vllm_engine.AutoModelForCausalLM') as mock_model:
                    mock_model.from_pretrained.return_value = Mock()
                    
                    engine = VLLMEngine(
                        model_name="gpt2",
                        tensor_parallel_size=1,
                        quantization=None
                    )
                    
                    assert engine.is_ready()
                    assert engine.model_name == "gpt2"
                    assert not engine.use_vllm
    
    @pytest.mark.asyncio
    async def test_generate_fallback(self):
        """Test generation with fallback engine."""
        from src.engine.vllm_engine import VLLMEngine
        
        with patch('src.engine.vllm_engine.VLLM_AVAILABLE', False):
            with patch('src.engine.vllm_engine.AutoTokenizer') as mock_tok:
                with patch('src.engine.vllm_engine.AutoModelForCausalLM') as mock_model:
                    with patch('src.engine.vllm_engine.pipeline') as mock_pipeline:
                        # Setup mocks
                        mock_tok.from_pretrained.return_value = Mock(
                            encode=Mock(return_value=[1, 2, 3]),
                            eos_token_id=0
                        )
                        mock_model.from_pretrained.return_value = Mock()
                        
                        # Mock pipeline output
                        mock_pipeline.return_value = Mock(
                            return_value=[{
                                "generated_text": "Test prompt and generated text"
                            }]
                        )
                        
                        engine = VLLMEngine(model_name="gpt2")
                        
                        result = await engine.generate(
                            prompt="Test prompt",
                            max_tokens=10
                        )
                        
                        assert "choices" in result
                        assert len(result["choices"]) > 0
                        assert "text" in result["choices"][0]
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics tracking."""
        from src.engine.vllm_engine import VLLMEngine
        
        with patch('src.engine.vllm_engine.VLLM_AVAILABLE', False):
            with patch('src.engine.vllm_engine.AutoTokenizer'):
                with patch('src.engine.vllm_engine.AutoModelForCausalLM'):
                    with patch('src.engine.vllm_engine.pipeline'):
                        engine = VLLMEngine(model_name="gpt2")
                        
                        stats = engine.get_stats()
                        assert "total_requests" in stats
                        assert stats["total_requests"] == 0

class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_metrics_initialization(self):
        """Test metrics collector initialization."""
        from src.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        assert collector is not None
    
    def test_record_latency(self):
        """Test latency recording."""
        from src.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_request()
        collector.record_latency(0.1)
        
        summary = collector.get_summary()
        assert "latency" in summary
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        from src.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_tokens(100)
        
        summary = collector.get_summary()
        assert "tokens" in summary

class TestConfiguration:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        from src.utils.config import Config
        
        config = Config()
        assert config.model_name is not None
        assert config.port == 8000
    
    def test_config_validation(self):
        """Test configuration validation."""
        from src.utils.config import Config
        
        config = Config()
        # Should not raise
        config._validate()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
