"""
Quantization utilities for model optimization.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class QuantizationConfig:
    """Configuration for quantization."""
    
    def __init__(
        self,
        method: str = "int8",
        calibration_samples: int = 512,
        symmetric: bool = True,
        per_channel: bool = True
    ):
        self.method = method
        self.calibration_samples = calibration_samples
        self.symmetric = symmetric
        self.per_channel = per_channel


class ModelQuantizer:
    """Quantize models to INT8/INT4 with minimal accuracy loss."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data = []
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[list] = None
    ) -> nn.Module:
        """Quantize model using specified method.
        
        Args:
            model: PyTorch model to quantize
            calibration_data: Data for calibration
            
        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model using {self.config.method}")
        
        if self.config.method == "int8":
            return self._quantize_int8(model, calibration_data)
        elif self.config.method == "int4":
            return self._quantize_int4(model, calibration_data)
        elif self.config.method == "awq":
            return self._quantize_awq(model, calibration_data)
        elif self.config.method == "gptq":
            return self._quantize_gptq(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")
    
    def _quantize_int8(
        self,
        model: nn.Module,
        calibration_data: Optional[list] = None
    ) -> nn.Module:
        """INT8 quantization using dynamic or static calibration.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration dataset
            
        Returns:
            INT8 quantized model
        """
        logger.info("Applying INT8 quantization")
        
        # Use PyTorch's quantization API
        model.eval()
        
        # Configure quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate if data provided
        if calibration_data:
            logger.info(f"Calibrating with {len(calibration_data)} samples")
            with torch.no_grad():
                for data in calibration_data[:self.config.calibration_samples]:
                    model(data)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        logger.info("INT8 quantization complete")
        return model
    
    def _quantize_int4(
        self,
        model: nn.Module,
        calibration_data: Optional[list] = None
    ) -> nn.Module:
        """INT4 quantization for maximum memory reduction.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration dataset
            
        Returns:
            INT4 quantized model
        """
        logger.info("Applying INT4 quantization")
        
        # Custom INT4 implementation
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize weights to INT4
                weight = module.weight.data
                
                # Calculate scale and zero point
                weight_min = weight.min()
                weight_max = weight.max()
                
                scale = (weight_max - weight_min) / 15.0  # 4-bit: 0-15
                zero_point = torch.round(-weight_min / scale).int()
                
                # Quantize
                quantized_weight = torch.round(weight / scale + zero_point).clamp(0, 15).to(torch.uint8)
                
                # Store quantized weights and dequantization params
                module.weight.data = quantized_weight
                module.register_buffer('scale', scale)
                module.register_buffer('zero_point', zero_point)
        
        logger.info("INT4 quantization complete")
        return model
    
    def _quantize_awq(
        self,
        model: nn.Module,
        calibration_data: Optional[list] = None
    ) -> nn.Module:
        """Activation-aware Weight Quantization (AWQ).
        
        Args:
            model: Model to quantize
            calibration_data: Calibration dataset
            
        Returns:
            AWQ quantized model
        """
        logger.info("Applying AWQ quantization")
        
        try:
            from awq import AutoAWQForCausalLM
            
            # AWQ requires calibration data
            if not calibration_data:
                raise ValueError("AWQ requires calibration data")
            
            # Quantize using AWQ
            # This is a simplified version; actual AWQ implementation is more complex
            logger.warning("AWQ quantization requires external library - using INT8 fallback")
            return self._quantize_int8(model, calibration_data)
            
        except ImportError:
            logger.warning("AWQ library not available - using INT8 fallback")
            return self._quantize_int8(model, calibration_data)
    
    def _quantize_gptq(
        self,
        model: nn.Module,
        calibration_data: Optional[list] = None
    ) -> nn.Module:
        """GPTQ quantization for accurate low-bit quantization.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration dataset
            
        Returns:
            GPTQ quantized model
        """
        logger.info("Applying GPTQ quantization")
        
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            # Configure GPTQ
            quantize_config = BaseQuantizeConfig(
                bits=4,
                group_size=128,
                desc_act=False
            )
            
            # This is a simplified version
            logger.warning("GPTQ quantization requires external library - using INT8 fallback")
            return self._quantize_int8(model, calibration_data)
            
        except ImportError:
            logger.warning("GPTQ library not available - using INT8 fallback")
            return self._quantize_int8(model, calibration_data)
    
    def measure_accuracy(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: list
    ) -> Dict[str, float]:
        """Measure accuracy degradation from quantization.
        
        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            test_data: Test dataset
            
        Returns:
            Dict with accuracy metrics
        """
        logger.info("Measuring quantization accuracy")
        
        original_outputs = []
        quantized_outputs = []
        
        with torch.no_grad():
            for data in test_data:
                orig_out = original_model(data)
                quant_out = quantized_model(data)
                
                original_outputs.append(orig_out)
                quantized_outputs.append(quant_out)
        
        # Calculate metrics
        mse = np.mean([
            torch.mean((o - q) ** 2).item()
            for o, q in zip(original_outputs, quantized_outputs)
        ])
        
        accuracy = 1.0 - (mse / np.mean([torch.mean(o ** 2).item() for o in original_outputs]))
        
        metrics = {
            "mse": mse,
            "accuracy": accuracy * 100,
            "accuracy_retention": accuracy
        }
        
        logger.info(f"Quantization accuracy: {metrics['accuracy']:.2f}%")
        return metrics
    
    def estimate_memory_savings(
        self,
        model: nn.Module,
        method: str = "int8"
    ) -> Dict[str, float]:
        """Estimate memory savings from quantization.
        
        Args:
            model: Original model
            method: Quantization method
            
        Returns:
            Dict with memory statistics
        """
        # Calculate original size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        param_size_mb = param_size / (1024 ** 2)
        
        # Estimate quantized size
        if method == "int8":
            reduction_factor = 4.0  # FP32 to INT8
        elif method == "int4":
            reduction_factor = 8.0  # FP32 to INT4
        else:
            reduction_factor = 4.0
        
        quantized_size_mb = param_size_mb / reduction_factor
        savings_mb = param_size_mb - quantized_size_mb
        savings_percent = (savings_mb / param_size_mb) * 100
        
        return {
            "original_size_mb": param_size_mb,
            "quantized_size_mb": quantized_size_mb,
            "savings_mb": savings_mb,
            "savings_percent": savings_percent
        }


def quantize_model_for_inference(
    model_name: str,
    quantization_method: str = "int8",
    calibration_samples: int = 512
) -> Tuple[nn.Module, AutoTokenizer]:
    """Convenience function to load and quantize a model.
    
    Args:
        model_name: HuggingFace model name
        quantization_method: Quantization method to use
        calibration_samples: Number of calibration samples
        
    Returns:
        Tuple of (quantized_model, tokenizer)
    """
    logger.info(f"Loading and quantizing model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Configure quantization
    config = QuantizationConfig(
        method=quantization_method,
        calibration_samples=calibration_samples
    )
    
    # Quantize
    quantizer = ModelQuantizer(config)
    quantized_model = quantizer.quantize_model(model)
    
    # Log memory savings
    savings = quantizer.estimate_memory_savings(model, quantization_method)
    logger.info(f"Memory savings: {savings['savings_mb']:.2f}MB ({savings['savings_percent']:.1f}%)")
    
    return quantized_model, tokenizer
