# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Callable

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.pytorch.quantization.fake_quant_builder import power_of_two_quantization, \
    symmetric_quantization, uniform_quantization
from model_compression_toolkit.core.pytorch.quantization.lut_fake_quant import activation_lut_kmean_quantizer


"""
Mapping from a QuantizationMethod to an activation quantizer function.
"""
_activation_quantizer_factory_mapping = {
    QuantizationMethod.POWER_OF_TWO: power_of_two_quantization,
    QuantizationMethod.SYMMETRIC: symmetric_quantization,
    QuantizationMethod.UNIFORM: uniform_quantization,
    QuantizationMethod.LUT_POT_QUANTIZER: activation_lut_kmean_quantizer
}


def get_activation_quantization_fn_factory(quantization_method: QuantizationMethod) -> Callable[[int, dict], Callable]:
    """
    Get factory for activation quantizer.

    Args:
        quantization_method: quantization method for activation.

    Returns:
        Factory that accepts activation bitwidth and a dict of quantization params, and returns the quantizer.
    """
    return _activation_quantizer_factory_mapping[quantization_method]
