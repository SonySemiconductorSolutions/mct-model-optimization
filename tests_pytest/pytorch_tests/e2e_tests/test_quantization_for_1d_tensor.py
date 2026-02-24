# Copyright 2026 Sony Semiconductor Solutions, Inc. All rights reserved.
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
import torch.nn as nn
import pytest

# This test checks whether an ActivationQuantizationHolder can be attached to a layer that produces 1D tensor output.
# These layers were selected from operators supported by the SDSP converter.

class Model(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.reshape(x, (-1,))

        if self.name == 'add':
            y = torch.add(x, 1)
        elif self.name == 'relu6':
            y = torch.nn.functional.relu6(x)
        elif self.name == 'relu':
            y = torch.nn.functional.relu(x)
        elif self.name == 'sigmoid':
            y = torch.nn.functional.sigmoid(x)
        elif self.name == 'leaky_relu':
            y = torch.nn.functional.leaky_relu(x)
        elif self.name == 'mul':
            y = torch.mul(x, 1)
        elif self.name == 'sub':
            y = torch.sub(x, 1)
        elif self.name == 'div':
            y = torch.div(x, 1)
        elif self.name == 'softmax':
            y = torch.nn.functional.softmax(x)
        elif self.name == 'tanh':
            y = torch.nn.functional.tanh(x)
        elif self.name == 'negative':
            y = torch.negative(x)
        elif self.name == 'abs':
            y = torch.abs(x)
        elif self.name == 'sqrt':
            y = torch.sqrt(torch.clamp(x, min=1e-6))
        elif self.name == 'rsqrt':
            y = torch.rsqrt(torch.clamp(x, min=1e-6))
        elif self.name == 'silu':
            y = torch.nn.functional.silu(x)
        elif self.name == 'hardswish':
            y = torch.nn.functional.hardswish(x)
        elif self.name == 'hardsigmoid':
            y = torch.nn.functional.hardsigmoid(x)
        elif self.name == 'pow':
            y = torch.pow(x, 1)
        elif self.name == 'gelu':
            y = torch.nn.functional.gelu(x)
        elif self.name == 'cos':
            y = torch.cos(x)
        elif self.name == 'sin':
            y = torch.sin(x)
        elif self.name == 'exp':
            y = torch.exp(x)
        elif self.name == 'mean':
            y = torch.mean(x, dim=0, keepdim=True)
        elif self.name == 'amax':
            y = torch.amax(x, dim=0, keepdim=True)
        elif self.name == 'maximum':
            y = torch.maximum(x, torch.tensor(0.0))
        elif self.name == 'minimum':
            y = torch.minimum(x, torch.tensor(0.0))
        elif self.name == 'sum':
            y = torch.sum(x, dim=0, keepdim=True)
        elif self.name == 'linalg_norm':
            y = torch.linalg.norm(x, dim=0, keepdim=True)
        return y

def representative_data_gen():
    yield [torch.randn(1, 3, 8, 8)]

@pytest.mark.parametrize("layer", [
    'add', 'relu6', 'relu', 'sigmoid', 'leaky_relu', 'mul', 'sub', 'div', 'mean', 'amax', 'softmax',
    'tanh', 'negative', 'maximum', 'minimum', 'abs', 'sqrt', 'sum', 'rsqrt', 'silu', 'hardswish', 'hardsigmoid',
    'linalg_norm', 'pow', 'gelu', 'cos', 'sin', 'exp', 
])
def test_ptq_1d_tensor(layer):

    float_model = Model(name=layer)

    tpc = mct.get_target_platform_capabilities("6.0")
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(float_model,
                                                                    representative_data_gen=representative_data_gen,
                                                                    target_platform_capabilities=tpc)
    
    if layer in ['abs', 'sum', 'pow']:
        activation_holder = f'{layer}_1_activation_holder_quantizer'
    else:
        activation_holder = f'{layer}_activation_holder_quantizer'

    assert hasattr(quantized_model, activation_holder)


@pytest.mark.parametrize("layer", [
    'add', 'relu6', 'relu', 'sigmoid', 'leaky_relu', 'mul', 'sub', 'div', 'mean', 'amax', 'softmax',
    'tanh', 'negative', 'maximum', 'minimum', 'abs', 'sqrt', 'sum', 'rsqrt', 'silu', 'hardswish', 'hardsigmoid',
    'linalg_norm', 'pow', 'gelu', 'cos', 'sin', 'exp', 
])
def test_ptq_mixed_precision_1d_tensor(layer):

    float_model = Model(name=layer)

    tpc = mct.get_target_platform_capabilities("6.0")
    core_config = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=1,
                                                                                                       use_hessian_based_scores=False))
    resource_utilization_data = mct.core.pytorch_resource_utilization_data(float_model,
                                                                           representative_data_gen,
                                                                           core_config,
                                                                           target_platform_capabilities=tpc)
    resource_utilization = mct.core.ResourceUtilization(resource_utilization_data.weights_memory * 0.9)
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(float_model,
                                                                    representative_data_gen,
                                                                    target_resource_utilization=resource_utilization,
                                                                    core_config=core_config,
                                                                    target_platform_capabilities=tpc)
    
    if layer in ['abs', 'sum', 'pow']:
        activation_holder = f'{layer}_1_activation_holder_quantizer'
    else:
        activation_holder = f'{layer}_activation_holder_quantizer'

    assert hasattr(quantized_model, activation_holder)


@pytest.mark.parametrize("layer", [
    'add', 'relu6', 'relu', 'sigmoid', 'leaky_relu', 'mul', 'sub', 'div', 'mean', 'amax', 'softmax',
    'tanh', 'negative', 'maximum', 'minimum', 'abs', 'sqrt', 'sum', 'rsqrt', 'silu', 'hardswish', 'hardsigmoid',
    'linalg_norm', 'pow', 'gelu', 'cos', 'sin', 'exp', 
])
def test_gptq_1d_tensor(layer):

    float_model = Model(name=layer)

    tpc = mct.get_target_platform_capabilities("6.0")
    gptq_config = mct.gptq.get_pytorch_gptq_config(n_epochs=5)
    quantized_model, _ = mct.gptq.pytorch_gradient_post_training_quantization(float_model,
                                                                              representative_data_gen,
                                                                              gptq_config=gptq_config,
                                                                              target_platform_capabilities=tpc)
    
    if layer in ['abs', 'sum', 'pow']:
        activation_holder = f'{layer}_1_activation_holder_quantizer'
    else:
        activation_holder = f'{layer}_activation_holder_quantizer'

    assert hasattr(quantized_model, activation_holder)