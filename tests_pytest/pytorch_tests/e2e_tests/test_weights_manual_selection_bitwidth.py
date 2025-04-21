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
import pytest

import model_compression_toolkit as mct
import torch
from torch.nn import Conv2d
from model_compression_toolkit.target_platform_capabilities.constants import BIAS, PYTORCH_KERNEL
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS
from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core import CoreConfig
from mct_quantizers import PytorchQuantizationWrapper

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig
from mct_quantizers import QuantizationMethod

kernel_weights_n_bits = 8
bias_weights_n_bits = 32
activation_n_bits = 8


def get_op_qco(kernel_n_bits, bias_n_bits):
    # define a default quantization config for all non-specified weights attributes.
    default_weight_attr_config = AttributeQuantizationConfig()

    # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
    kernel_base_config = AttributeQuantizationConfig(
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True)

    base_cfg = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO
    )

    mx_cfg_list = [base_cfg]
    if kernel_weights_n_bits is not None:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: kernel_n_bits}}))
    if bias_weights_n_bits is not None:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: bias_n_bits}}))

    return base_cfg, mx_cfg_list, default_config


def generate_tpc_local(default_config, base_config, mixed_precision_cfg_list):
    default_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple(mixed_precision_cfg_list),
        base_config=base_config)

    operator_set = []

    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
    add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD)
    operator_set.extend([conv, relu, add])

    generated_tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set))

    return generated_tpc


def get_tpc(kernel_n_bits, bias_n_bits):
    base_cfg, mx_cfg_list, default_config = get_op_qco(kernel_n_bits, bias_n_bits)
    tpc = generate_tpc_local(default_config, base_cfg, mx_cfg_list)
    return tpc

def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

def get_float_model():
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                return x
        return BaseModel()


class TestManualWeightsBitwidthSelectionByLayerType:
    # (LayerType, bit width, attribute, kernel_n_bits, bias_n_bits)
    test_input_1 = (NodeTypeFilter(Conv2d), 16, PYTORCH_KERNEL, 16, None)
    test_input_2 = (NodeTypeFilter(Conv2d), [2], [PYTORCH_KERNEL], 2, None)
    
    test_expected_1 = ([Conv2d], [16])
    test_expected_2 = ([Conv2d], [2])
    
    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
    ])

    def test_manual_weights_bitwidth_selection(self, inputs, expected):
        float_model = get_float_model()

        target_platform_cap = get_tpc(kernel_n_bits=inputs[3], bias_n_bits=inputs[4])
        
        core_config = CoreConfig()
        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])
        
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )

        for name, layer in quantized_model.named_modules():
            # check if the layer is a weights quantizer
            if isinstance(layer, PytorchQuantizationWrapper):
                expected_bitwidths = expected[1]
                attrs = inputs[2]

                if not isinstance(attrs, list):
                    attrs = [attrs]

                for bitwidth, attr in zip(expected_bitwidths, attrs):
                    
                    if layer.weights_quantizers.get(attr) is not None:
                        assert layer.weights_quantizers.get(attr).num_bits == bitwidth


class TestManualWeightsBitwidthSelectionByLayerName:
    # (LayerName, bit width, attribute, kernel_n_bits, bias_n_bits)
    test_input_1 = (NodeNameFilter("conv1"), 16, PYTORCH_KERNEL, 16, None)
    test_input_2 = (NodeNameFilter("conv1"), [2], [PYTORCH_KERNEL], 2, None)
    test_input_3 = ([NodeNameFilter("conv1"), NodeNameFilter("conv1")], [4, 16], [PYTORCH_KERNEL, BIAS], 4, 16)

    test_expected_1 = (["conv1"], [16])
    test_expected_2 = (["conv1"], [2])
    test_expected_3 = (["conv1", "conv1"], [4, 16])
    
    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])

    def test_manual_weights_bitwidth_selection(self, inputs, expected):

        float_model = get_float_model()

        target_platform_cap = get_tpc(kernel_n_bits=inputs[3], bias_n_bits=inputs[4])
        
        core_config = CoreConfig()
        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])
        
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )

        for name, layer in quantized_model.named_modules():
            # check if the layer is a weights quantizer
            if isinstance(layer, PytorchQuantizationWrapper):
                bitwidths = expected[1]
                attrs = inputs[2]

                if name == "conv1":
                    for attr, bitwidth in zip(attrs, bitwidths):
                        if layer.weights_quantizers.get(attr) is not None:
                            assert layer.weights_quantizers.get(attr).num_bits == bitwidth
                else:
                    for attr in attrs:
                        if layer.weights_quantizers.get(attr) is not None:
                                if attr == PYTORCH_KERNEL:
                                    assert layer.weights_quantizers.get(attr).num_bits == kernel_weights_n_bits
                                elif attr == BIAS:
                                    assert layer.weights_quantizers.get(attr).num_bits == bias_weights_n_bits
