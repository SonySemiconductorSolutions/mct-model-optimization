# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from typing import Iterator, List
import torch
import torch.nn as nn
import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from mct_quantizers import QuantizationMethod


class TakeModel(nn.Module):

    def __init__(self, indices):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def forward(self, x):
        x = self.relu(self.conv(x))
        output = torch.take(x, self.indices)
        return output


def get_representative_dataset(n_iter=1):

    def representative_dataset() -> Iterator[List]:
        for _ in range(n_iter):
            yield [torch.randn(1, 3, 32, 32)]
    return representative_dataset


def get_edgemdt_tpc_v6():

    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=schema.Signedness.AUTO)

    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
    dim_manipulation_config = (default_configuration_options.clone_and_edit(enable_activation_quantization=False,
                                                                            quantization_preserving=True,
                                                                            supported_input_activation_n_bits=(8, 16))
                               .clone_and_edit_weight_attribute(enable_weights_quantization=False))
    operator_set = []
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.TAKE, qc_options=dim_manipulation_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=default_configuration_options))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.RELU, qc_options=default_configuration_options))

    tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set))
    return tpc


def test_take():

    model = TakeModel(indices=[0, 100])
    tpc = get_edgemdt_tpc_v6() # TPC equivalent to edgemdt-tpc v6.0
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(model, 
                                                                    get_representative_dataset(n_iter=1),
                                                                    target_resource_utilization=None,
                                                                    core_config=mct.core.CoreConfig(),
                                                                    target_platform_capabilities=tpc)
    assert hasattr(quantized_model, 'take')
    assert not hasattr(quantized_model, 'take_activation_holder_quantizer')