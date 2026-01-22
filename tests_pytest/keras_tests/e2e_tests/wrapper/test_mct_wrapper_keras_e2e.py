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
End-to-End Testing for MCTWrapper with Keras/TensorFlow Framework

This module provides comprehensive end-to-end tests for the MCTWrapper
quantization functionality using Keras/TensorFlow models. It tests various
quantization methods including PTQ, GPTQ, and their mixed-precision variants.

Test Coverage:
- Post-Training Quantization (PTQ)
- PTQ with Mixed Precision
- Gradient Post-Training Quantization (GPTQ)
- GPTQ with Mixed Precision

The tests use a simple CNN model and random data for representative
dataset generation for quantization testing.
"""
import pytest
import tensorflow as tf
import keras
from typing import Callable, List, Any, Tuple, Iterator
import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationErrorMethod

@pytest.fixture
def get_model():
    """
    Create a simple CNN model for Keras/TensorFlow quantization testing.
    
    Returns:
        keras.Model: Simple CNN model for testing
    """
    inputs = keras.Input(shape=(32, 32, 3))
    x1 = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x2 = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    outputs = keras.layers.Concatenate(axis=-1)([x1, x2])
    return keras.Model(inputs, outputs)

@pytest.fixture
def get_representative_dataset(n_iter=5):
    """
    Create representative dataset generator for Keras/TensorFlow quantization.
    
    Returns:
        function: Generator function that yields batches of random data
    """
    def representative_dataset() -> Iterator[List]:
        for _ in range(n_iter):
            yield [tf.random.normal((1, 32, 32, 3)).numpy()]
    return representative_dataset

@pytest.mark.parametrize("quant_func", [
    "PTQ_Keras",
    "PTQ_Keras_mixed_precision",
    "GPTQ_Keras",
    "GPTQ_Keras_mixed_precision"
])
def test_quantization(
        quant_func: str,
        get_model: Callable[[], keras.Model],
        get_representative_dataset: Callable[[], Iterator[List[Any]]]
        ) -> None:
    """
    Test end-to-end quantization workflows for Keras/TensorFlow models.
    
    Args:
        quant_func (str): Name of quantization function to test
        get_model: Fixture providing simple CNN model
        get_representative_dataset: Fixture providing representative data
        
    Test Methods:
        - PTQ_Keras: Standard Post-Training Quantization
        - PTQ_Keras_mixed_precision: PTQ with Mixed Precision optimization
        - GPTQ_Keras: Gradient-based Post-Training Quantization
        - GPTQ_Keras_mixed_precision: GPTQ with Mixed Precision optimization
    """
    
    # Get model and representative dataset using fixtures
    float_model = get_model
    representative_dataset_gen = get_representative_dataset

    # Decorator to print logs before and after function execution
    def decorator(func: Callable[[keras.Model], Tuple[bool, keras.Model]]) -> Callable[[keras.Model], Tuple[bool, keras.Model]]:
        """
        Decorator for logging quantization function execution.
        
        This decorator wraps quantization functions to provide clear logging
        of when each quantization method starts and ends, and handles any
        failures by terminating execution.
        
        Args:
            func: Quantization function to wrap
            
        Returns:
            function: Wrapped function with logging capabilities
        """
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[bool, keras.Model]:
            print(f"----------------- {func.__name__} Start ---------------")
            flag, result = func(*args, **kwargs)
            print(f"----------------- {func.__name__} End -----------------")
            if not flag:
                exit()
            return flag, result
        return wrapper

    #########################################################################
    # Run PTQ (Post-Training Quantization) with Keras
    @decorator
    def PTQ_Keras(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        """
        Execute Post-Training Quantization using MCT Target Platform Capabilities.
        
        This method applies standard PTQ without mixed precision, using MCT's
        predefined target platform capabilities for optimal quantization settings.
        """
        # Quantization method configuration
        framework = 'tensorflow'
        method = 'PTQ'
        use_mixed_precision = False

        # Configure quantization parameters for optimal model performance
        param_items = [
            ['sdsp_version', '3.14'],  # The version of the SDSP converter.
            ['activation_error_method', QuantizationErrorMethod.MSE],  # ErrorMethod.
            ['weights_bias_correction', True],  # Enable bias correction
            ['z_threshold', float('inf')],  # Z threshold
            ['linear_collapsing', True],  # Enable linear collapsing
            ['residual_collapsing', True],  # Enable residual collapsing
            ['save_model_path', './qmodel_PTQ_Keras.keras']  # Path to save the model.
        ]

        # Execute quantization using MCTWrapper
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, representative_dataset_gen, framework, method, use_mixed_precision, param_items)
        return flag, quantized_model

    #########################################################################
    # Run PTQ + Mixed Precision Quantization with Keras
    @decorator
    def PTQ_Keras_mixed_precision(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        """
        Execute PTQ with Mixed Precision optimization for better accuracy.
        
        Mixed Precision allows different layers to use different bit-widths,
        optimizing the trade-off between model size and accuracy.
        """
        # Quantization method configuration
        framework = 'tensorflow'
        method = 'PTQ'
        use_mixed_precision = True

        # Configure mixed precision parameters for optimal compression
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
            ['save_model_path', './qmodel_PTQ_Keras_mixed_precision.keras']  # Path to save the quantized model.
        ]

        # Execute quantization with mixed precision using MCTWrapper
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, representative_dataset_gen, framework, method, use_mixed_precision, param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ (Gradient-based PTQ) with Keras
    @decorator
    def GPTQ_Keras(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        """
        Execute Gradient-based Post-Training Quantization for enhanced accuracy.
        
        GPTQ uses gradient information to fine-tune quantization parameters,
        resulting in better model accuracy compared to standard PTQ.
        """
        # Quantization method configuration
        framework = 'tensorflow'
        method = 'GPTQ'
        use_mixed_precision = False

        # Configure GPTQ-specific parameters for gradient-based optimization
        param_items = [
            ['sdsp_version', '3.14'],  # The version of the SDSP converter.
            ['activation_error_method', QuantizationErrorMethod.MSE],  # Error metric for activation (low priority).
            ['weights_bias_correction', True],  # Enable bias correction for weights (low priority).
            ['z_threshold', float('inf')],  # Threshold for zero-point quantization (low priority).
            ['linear_collapsing', True],  # Enable linear layer collapsing optimization (low priority).
            ['residual_collapsing', True],  # Enable residual connection collapsing (low priority).    
            ['n_epochs', 5],  # Number of epochs for gradient-based fine-tuning.
            ['optimizer', None],  # Optimizer to use during fine-tuning (low priority).
            ['save_model_path', './qmodel_GPTQ_Keras.keras']  # Path to save the quantized model.
        ]

        # Execute gradient-based quantization using MCTWrapper
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, representative_dataset_gen, framework, method, use_mixed_precision, param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ + Mixed Precision Quantization (mixed_precision) with Keras
    @decorator
    def GPTQ_Keras_mixed_precision(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        framework = 'tensorflow'
        method = 'GPTQ'
        use_mixed_precision = True

        param_items = [
            ['sdsp_version', '3.14'],  # The version of the SDSP converter.
            ['activation_error_method', QuantizationErrorMethod.MSE],  # Error metric for activation (low priority).
            ['weights_bias_correction', True],  # Enable bias correction for weights (low priority).
            ['z_threshold', float('inf')],  # Threshold for zero-point quantization (low priority).
            ['linear_collapsing', True],  # Enable linear layer collapsing optimization (low priority).
            ['residual_collapsing', True],  # Enable residual connection collapsing (low priority).
            ['n_epochs', 5],  # Number of epochs for gradient-based fine-tuning.
            ['optimizer', None],  # Optimizer to use during fine-tuning (low priority).
            ['distance_weighting_method', None],  # Distance weighting method for GPTQ (low priority).
            ['num_of_images', 5],  # Number of images to use for calibration.
            ['use_hessian_based_scores', False],  # Whether to use Hessian-based scores for layer importance (low priority).
            ['weights_compression_ratio', 0.75],  # Compression ratio for weights.
            ['save_model_path', './qmodel_GPTQ_Keras_mixed_precision.keras']  # Path to save the quantized model.
        ]

        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, representative_dataset_gen, framework, method, use_mixed_precision, param_items)
        return flag, quantized_model

    #########################################################################
    # Run LQPTQ (Low-bit Quantizer PTQ) with Keras
    @decorator
    def LQPTQ_Keras(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        framework = 'tensorflow'
        method = 'LQPTQ'
        use_mixed_precision = False

        param_items = [
            ['learning_rate', 0.0001],  # Learning rate
            ['converter_ver', 'v3.14'],  # Converter version
            ['save_model_path', './qmodel_LQPTQ_Keras.keras']  # Path to save the model.
        ]

        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, representative_dataset_gen, framework, method, use_mixed_precision, param_items)
        return flag, quantized_model

    # Execute the selected quantization method
    quant_methods = {
        "PTQ_Keras": PTQ_Keras,
        "PTQ_Keras_mixed_precision": PTQ_Keras_mixed_precision,
        "GPTQ_Keras": GPTQ_Keras,
        "GPTQ_Keras_mixed_precision": GPTQ_Keras_mixed_precision
    }
    
    # Run the selected quantization method and verify success
    flag, quantized_model = quant_methods[quant_func](float_model)
    assert flag, f"Quantization failed for method: {quant_func}"

    # Success confirmation
    print(f"{quant_func} quantization completed successfully!")
