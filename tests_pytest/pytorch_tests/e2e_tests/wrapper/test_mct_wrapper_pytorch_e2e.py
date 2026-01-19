#  Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

"""
End-to-End Testing for MCTWrapper with PyTorch Framework

This module provides comprehensive end-to-end tests for the MCTWrapper
quantization functionality using PyTorch models. It tests various
quantization methods including PTQ, GPTQ, and their mixed-precision variants.

Test Coverage:
- Post-Training Quantization (PTQ)
- PTQ with Mixed Precision
- Gradient Post-Training Quantization (GPTQ)
- GPTQ with Mixed Precision

The tests use a simple CNN model and random data for representative
dataset generation for quantization testing. All quantized models are
exported to ONNX format for cross-platform deployment.
"""
import pytest
import torch
import torch.nn as nn
from typing import Callable, List, Tuple, Any, Iterator
import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationErrorMethod

@pytest.fixture
def get_model():

    class StackModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU()
            )

        def forward(self, x):
            out1 = self.conv1(x)
            out2 = self.conv2(x)
            output = torch.stack([out1, out2], dim=-1)
            return output
    return StackModel()

@pytest.fixture
def get_representative_dataset(n_iter=5):

    def representative_dataset() -> Iterator[List]:
        for _ in range(n_iter):
            # Force CPU tensor creation
            yield [torch.randn(1, 3, 32, 32, device='cpu')]
    return representative_dataset

@pytest.mark.parametrize("quant_func", [
    "PTQ_Pytorch",
    "PTQ_Pytorch_mixed_precision",
    "GPTQ_Pytorch",
    "GPTQ_Pytorch_mixed_precision",
])
def test_quantization(
        quant_func: str,
        get_model: Callable[[], torch.nn.Module],
        get_representative_dataset: Callable[[], Iterator[List[torch.Tensor]]]
        ) -> None:
    """
    Test end-to-end quantization workflows for PyTorch models.
    
    This comprehensive test function validates different PyTorch quantization
    methods by executing the complete workflow from model preparation through
    quantization to accuracy evaluation and ONNX export.
 
    Args:
        quant_func (str): Name of quantization method to test
        
    Test Methods:
        - PTQ_Pytorch: Standard Post-Training Quantization
        - PTQ_Pytorch_mixed_precision: PTQ with Mixed Precision optimization
        - GPTQ_Pytorch: Gradient-based Post-Training Quantization
        - GPTQ_Pytorch_mixed_precision: GPTQ with Mixed Precision optimization
        
    Export Format:
        All quantized models are exported to ONNX format for cross-platform
        deployment and inference optimization.
    """
    
    # Get model and representative dataset using fixtures
    float_model = get_model
    representative_dataset_gen = get_representative_dataset
    
    # Decorator to print logs before and after function execution
    def decorator(func: Callable[[torch.nn.Module], Tuple[bool, torch.nn.Module]]) -> Callable[[torch.nn.Module], Tuple[bool, torch.nn.Module]]:
        """
        Decorator for logging quantization function execution.
        
        This decorator wraps quantization functions to provide clear logging
        of execution progress and proper error handling. It tracks when each
        quantization method starts and ends, and converts failures into
        clear runtime errors.
        
        Args:
            func: Quantization function to wrap
            
        Returns:
            function: Wrapped function with logging and error handling
        """
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[bool, torch.nn.Module]:
            print(f"----------------- {func.__name__} Start ---------------")
            flag, result = func(*args, **kwargs)
            print(f"----------------- {func.__name__} End -----------------")
            if not flag:
                raise RuntimeError(f"Quantization failed for {func.__name__}")
            return flag, result
        return wrapper

    #########################################################################
    # Run PTQ (Post-Training Quantization) with PyTorch
    @decorator
    def PTQ_Pytorch(float_model):
        """
        Execute Post-Training Quantization (PTQ) on PyTorch model.
        
        PTQ is a quantization method that converts a pre-trained floating-point
        PyTorch model to a quantized model without requiring additional
        training.
        It uses representative data to determine optimal quantization
        parameters
        and exports the result to ONNX format.
        
        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        # Configure quantization framework and method settings
        framework = 'pytorch'
        method = 'PTQ'
        use_mixed_precision = False  # Disable mixed precision for standard PTQ

        # Define quantization parameters for optimal model performance
        param_items = [
            ['sdsp_version', '3.14'],  # The version of the SDSP converter.
            ['activation_error_method', QuantizationErrorMethod.MSE],  # Error method
            ['weights_bias_correction', True],  # Enable bias correction
            ['z_threshold', float('inf')],  # Z threshold
            ['linear_collapsing', True],  # Enable linear collapsing
            ['residual_collapsing', True],  # Enable residual collapsing
            ['save_model_path', './qmodel_PTQ_Pytorch.onnx']  # Path to save the model.
        ]

        # Execute quantization using MCTWrapper and export to ONNX
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, representative_dataset_gen, framework, method, use_mixed_precision,
            param_items)
        return flag, quantized_model

    #########################################################################
    # Run PTQ + Mixed Precision Quantization with PyTorch
    @decorator
    def PTQ_Pytorch_mixed_precision(float_model):
        """
        Execute PTQ with Mixed Precision Quantization on PyTorch model.
        
        This method combines Post-Training Quantization with Mixed Precision
        optimization to achieve better accuracy-efficiency trade-offs. It
        automatically determines optimal bit-width allocation for different
        layers based on their sensitivity to quantization.
        
        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        # Configure quantization method with mixed precision enabled
        framework = 'pytorch'
        method = 'PTQ'
        use_mixed_precision = True      # Enable mixed precision optimization

        # Define mixed precision quantization parameters
        param_items = [
            ['sdsp_version', '3.14'],  # The version of the SDSP converter.
            ['activation_error_method', QuantizationErrorMethod.MSE],  # Error metric for activation (low priority).
            ['weights_bias_correction', True],  # Enable bias correction for weights (low priority).
            ['z_threshold', float('inf')],  # Threshold for zero-point quantization (low priority).
            ['linear_collapsing', True],  # Enable linear layer collapsing optimization (low priority).
            ['residual_collapsing', True],  # Enable residual connection collapsing (low priority).
            ['distance_weighting_method', None],  # Distance weighting method for mixed precision (low priority).
            ['num_of_images', 5],  # Number of images for mixed precision.
            ['use_hessian_based_scores', False],  # Use Hessian-based sensitivity scores for layer importance (low priority).
            ['weights_compression_ratio', 0.75],  # Target compression ratio for model weights (75% of original size).
            ['save_model_path', './qmodel_PTQ_Pytorch_mixed_precision.onnx']  # Path to save quantized model as ONNX.
        ]

        # Execute mixed precision quantization and export to ONNX
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, representative_dataset_gen, framework, method, use_mixed_precision,
            param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ (Gradient-based PTQ) with PyTorch
    @decorator
    def GPTQ_Pytorch(float_model):
        """
        Execute Gradient-based Post-Training Quantization (GPTQ) on
        PyTorch model.
        
        GPTQ is an advanced quantization method that uses gradient-based
        optimization to fine-tune quantization parameters. It iteratively
        adjusts the quantized weights to minimize the loss function, resulting
        in better accuracy preservation compared to standard PTQ.
 
        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        # Configure gradient-based quantization method
        framework = 'pytorch'
        method = 'GPTQ'
        use_mixed_precision = False     # Disable mixed precision for standard GPTQ

        # Define GPTQ-specific parameters for gradient-based optimization
        param_items = [
            ['sdsp_version', '3.14'],  # The version of the SDSP converter.
            ['activation_error_method', QuantizationErrorMethod.MSE],  # Error metric for activation (low priority).
            ['weights_bias_correction', True],  # Enable bias correction for weights (low priority).
            ['z_threshold', float('inf')],  # Threshold for zero-point quantization (low priority).
            ['linear_collapsing', True],  # Enable linear layer collapsing optimization (low priority).
            ['residual_collapsing', True],  # Enable residual connection collapsing (low priority).    
            ['n_epochs', 5],  # Number of epochs for gradient-based fine-tuning.
            ['optimizer', None],  # Optimizer to use during fine-tuning (low priority).
            ['save_model_path', './qmodel_GPTQ_Pytorch.onnx']  # Path to save quantized model as ONNX.
        ]

        # Execute gradient-based quantization and export to ONNX
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, representative_dataset_gen, framework, method, use_mixed_precision,
            param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ + Mixed Precision Quantization with PyTorch
    @decorator
    def GPTQ_Pytorch_mixed_precision(float_model):
        """
        Execute GPTQ with Mixed Precision Quantization on PyTorch model.
        
        This method combines Gradient-based Post-Training Quantization with
        Mixed Precision optimization to achieve the best possible accuracy-
        efficiency trade-off. It uses gradient-based fine-tuning while
        automatically selecting optimal bit-widths for different layers.

        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        # Configure gradient-based quantization with mixed precision
        framework = 'pytorch'
        method = 'GPTQ'
        use_mixed_precision = True      # Enable mixed precision for optimal accuracy

        # Define GPTQ mixed precision parameters for advanced optimization
        param_items = [
            ['sdsp_version', '3.14'],  # The version of the SDSP converter.
            ['activation_error_method', QuantizationErrorMethod.MSE],  # Error metric for activation (low priority).
            ['weights_bias_correction', True],  # Enable bias correction for weights (low priority).
            ['z_threshold', float('inf')],  # Threshold for zero-point quantization (low priority).
            ['linear_collapsing', True],  # Enable linear layer collapsing optimization (low priority).
            ['residual_collapsing', True],  # Enable residual connection collapsing (low priority).
            ['weights_compression_ratio', 0.75],  # Compression ratio for weights.
            ['n_epochs', 5],  # Number of epochs for gradient-based fine-tuning.
            ['optimizer', None],  # Optimizer to use during fine-tuning (low priority).
            ['distance_weighting_method', None],  # Distance weighting method for GPTQ (low priority).
            ['num_of_images', 5],  # Number of images to use for calibration.
            ['use_hessian_based_scores', False],  # Whether to use Hessian-based scores for layer importance (low priority).
            ['save_model_path', './qmodel_GPTQ_Pytorch_mixed_precision.onnx']  # Path to save quantized model as ONNX.
        ]

        # Execute advanced GPTQ with mixed precision and export to ONNX
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, representative_dataset_gen, framework, method, use_mixed_precision,
            param_items)
        return flag, quantized_model

    # Execute the selected quantization method
    quant_methods = {
        "PTQ_Pytorch": PTQ_Pytorch,
        "PTQ_Pytorch_mixed_precision": PTQ_Pytorch_mixed_precision,
        "GPTQ_Pytorch": GPTQ_Pytorch,
        "GPTQ_Pytorch_mixed_precision": GPTQ_Pytorch_mixed_precision,
    }
    
    # Run the quantization method and verify successful completion
    flag, quantized_model = quant_methods[quant_func](float_model)
    assert flag, f"Quantization failed for {quant_func}"

    # Success confirmation
    print(f"{quant_func} quantization completed successfully!")
