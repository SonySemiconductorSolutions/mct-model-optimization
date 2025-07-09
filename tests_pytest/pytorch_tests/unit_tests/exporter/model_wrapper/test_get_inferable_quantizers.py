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
from unittest.mock import Mock

from model_compression_toolkit.exporter.model_wrapper.fw_agnostic.get_inferable_quantizers import get_inferable_quantizers
from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.activation_pot_inferable_quantizer import ActivationPOTInferableQuantizer


def test_get_inferable_quantizers():

    # QUANT node
    node_mock = Mock()
    node_mock.is_activation_quantization_enabled.return_value = True
    node_mock.is_fln_quantization.return_value = False
    node_mock.output_shape = 1

    get_activations_quantizer_for_node_mock = Mock()
    get_activations_quantizer_for_node_mock.return_value = ActivationPOTInferableQuantizer(num_bits=8, signed=True, threshold=[8.0])

    _, activation_quantizers = get_inferable_quantizers(node=node_mock, get_weights_quantizer_for_node=None, 
                                                        get_activations_quantizer_for_node=get_activations_quantizer_for_node_mock)
    
    assert len(activation_quantizers) == 1
    assert isinstance(activation_quantizers[0], ActivationPOTInferableQuantizer)
    assert activation_quantizers[0].num_bits == 8
    assert activation_quantizers[0].signed == True
    assert activation_quantizers[0].threshold_np == 8.0
    
    # FLN_QUANT node
    node_mock = Mock()
    node_mock.is_activation_quantization_enabled.return_value = False
    node_mock.is_fln_quantization.return_value = True
    node_mock.output_shape = 1

    get_activations_quantizer_for_node_mock = Mock()
    get_activations_quantizer_for_node_mock.return_value = ActivationPOTInferableQuantizer(num_bits=16, signed=False, threshold=[4.0])
    _, activation_quantizers = get_inferable_quantizers(node=node_mock, get_weights_quantizer_for_node=None, 
                                                        get_activations_quantizer_for_node=get_activations_quantizer_for_node_mock)
    
    assert len(activation_quantizers) == 1
    assert isinstance(activation_quantizers[0], ActivationPOTInferableQuantizer)
    assert activation_quantizers[0].num_bits == 16
    assert activation_quantizers[0].signed == False
    assert activation_quantizers[0].threshold_np == 4.0

    # NO_QUANT and FLN_NO_QUANT node
    node_mock = Mock()
    node_mock.is_activation_quantization_enabled.return_value = False
    node_mock.is_fln_quantization.return_value = False
    node_mock.output_shape = 1

    get_activations_quantizer_for_node_mock = Mock()
    get_activations_quantizer_for_node_mock.return_value = ActivationPOTInferableQuantizer(num_bits=8, signed=True, threshold=[16.0])
    _, activation_quantizers = get_inferable_quantizers(node=node_mock, get_weights_quantizer_for_node=None, 
                                                        get_activations_quantizer_for_node=get_activations_quantizer_for_node_mock)
    
    assert len(activation_quantizers) == 0
