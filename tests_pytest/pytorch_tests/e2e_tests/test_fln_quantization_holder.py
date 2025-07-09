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
import model_compression_toolkit as mct
import torch
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchFLNActivationQuantizationHolder

from tests_pytest._test_util.tpc_util import configure_mp_activation_opsets
from model_compression_toolkit.target_platform_capabilities.schema.v2 import QuantizationMethod, AttributeQuantizationConfig, \
    OpQuantizationConfig, QuantizationConfigOptions, Signedness, OperatorSetNames, TargetPlatformCapabilities, Fusing, OperatorsSet
from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc


def build_tpc():
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=[8],
        enable_activation_quantization=True,
        enable_weights_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO
    )

    opsets, _ = configure_mp_activation_opsets(
        opset_names=[OperatorSetNames.CONV,
                     OperatorSetNames.RELU,
                     OperatorSetNames.SIGMOID,
                     OperatorSetNames.FULLY_CONNECTED,
                     OperatorSetNames.HARDSWISH],
        base_op_config=default_op_cfg,
        a_nbits=[8]
    )
    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])

    test_qc = generate_test_op_qc(**generate_test_attr_configs(), activation_n_bits=16)

    tpc = TargetPlatformCapabilities(
        default_qco=default_cfg,
        operator_set=opsets,
        fusing_patterns=[
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.CONV),
            OperatorsSet(name=OperatorSetNames.RELU)), fuse_op_quantization_config=test_qc),
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.CONV),
            OperatorsSet(name=OperatorSetNames.SIGMOID))),
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.FULLY_CONNECTED),
            OperatorsSet(name=OperatorSetNames.HARDSWISH)), fuse_op_quantization_config=test_qc),
        ]
    )
    return tpc

def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

def get_float_model():
    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.sigmoid = torch.nn.Sigmoid()
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(in_features=48, out_features=10)
            self.hswish = torch.nn.Hardswish()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.sigmoid(x)
            x = self.flatten(x)
            x = self.fc(x)
            x = self.hswish(x)
            return x
    return BaseModel()

def test_fln_quantization_holder():

    float_model = get_float_model()
    tpc = build_tpc()

    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_data_gen,
        target_platform_capabilities=tpc
    )

    # check conv1
    assert hasattr(quantized_model, 'conv1_activation_holder_quantizer')
    conv1_activation_holder_quantizer = quantized_model.conv1_activation_holder_quantizer
    assert isinstance(conv1_activation_holder_quantizer, PytorchFLNActivationQuantizationHolder)
    assert conv1_activation_holder_quantizer.quantization_bypass == True
    assert conv1_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16

    # check relu
    assert hasattr(quantized_model, 'relu_activation_holder_quantizer')
    relu_activation_holder_quantizer = quantized_model.relu_activation_holder_quantizer
    assert isinstance(relu_activation_holder_quantizer, PytorchActivationQuantizationHolder)
    assert relu_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8

    # check conv2
    assert not hasattr(quantized_model, 'conv2_activation_holder_quantizer')
    
    # check sigmoid
    assert hasattr(quantized_model, 'sigmoid_activation_holder_quantizer')
    sigmoid_activation_holder_quantizer = quantized_model.sigmoid_activation_holder_quantizer
    assert isinstance(sigmoid_activation_holder_quantizer, PytorchActivationQuantizationHolder)
    assert sigmoid_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8

    # check fc
    assert hasattr(quantized_model, 'fc_activation_holder_quantizer')
    fc_activation_holder_quantizer = quantized_model.fc_activation_holder_quantizer
    assert isinstance(fc_activation_holder_quantizer, PytorchFLNActivationQuantizationHolder)
    assert fc_activation_holder_quantizer.quantization_bypass == True
    assert fc_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16

    # check hswish
    assert hasattr(quantized_model, 'hswish_activation_holder_quantizer')
    hswish_activation_holder_quantizer = quantized_model.hswish_activation_holder_quantizer
    assert isinstance(hswish_activation_holder_quantizer, PytorchActivationQuantizationHolder)
    assert hswish_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8