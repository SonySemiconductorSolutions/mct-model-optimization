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
import numpy as np
import model_compression_toolkit as mct
import torch
import torch.nn as nn
import pytest


class ScalarModel(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.scalar = nn.Parameter(2.0 * torch.ones([])) # Scalar

    def forward(self, x):
        if self.name == 'add':
            const = torch.add(self.scalar, 1)
        elif self.name == 'relu6':
            const = torch.nn.functional.relu6(self.scalar)
        elif self.name == 'relu':
            const = torch.relu(self.scalar)
        elif self.name == 'sigmoid':
            const = torch.sigmoid(self.scalar)
        elif self.name == 'eq':
            const = torch.eq(self.scalar, 1)
        elif self.name == 'leaky_relu':
            const = torch.nn.functional.leaky_relu(self.scalar)
        elif self.name == 'mul':
            const = torch.mul(self.scalar, 1)
        elif self.name == 'sub':
            const = torch.sub(self.scalar, 1)
        elif self.name == 'div':
            const = torch.div(self.scalar, 1)
        elif self.name == 'softmax':
            const = torch.nn.functional.softmax(self.scalar)
        elif self.name == 'tanh':
            const = torch.tanh(self.scalar)
        elif self.name == 'negative':
            const = torch.negative(self.scalar)
        elif self.name == 'abs':
            const = torch.abs(self.scalar)
        elif self.name == 'sqrt':
            const = torch.sqrt(self.scalar)
        elif self.name == 'sum':
            const = torch.sum(self.scalar)
        elif self.name == 'rsqrt':
            const = torch.rsqrt(self.scalar)
        elif self.name == 'silu':
            const = torch.nn.functional.silu(self.scalar)
        elif self.name == 'hardswish':
            const = torch.nn.functional.hardswish(self.scalar)
        elif self.name == 'hardsigmoid':
            const = torch.nn.functional.hardsigmoid(self.scalar)
        elif self.name == 'pow':
            const = torch.pow(self.scalar, 1)
        elif self.name == 'gelu':
            const = torch.nn.functional.gelu(self.scalar)
        elif self.name == 'cos':
            const = torch.cos(self.scalar)
        elif self.name == 'sin':
            const = torch.sin(self.scalar)
        elif self.name == 'exp':
            const = torch.exp(self.scalar)
        
        y = x + const
        return y

def representative_data_gen():
    yield [np.random.random((1, 3, 8, 8))]

@pytest.mark.parametrize("layer_name", [
    'add', 'relu6', 'relu', 'sigmoid', 'eq', 'leaky_relu', 'mul', 'sub', 'div', 'softmax',
    'tanh', 'negative', 'abs', 'sqrt', 'sum', 'rsqrt', 'silu', 'hardswish', 'hardsigmoid',
    'pow', 'gelu', 'cos', 'sin', 'exp'
])
def test_scalar_layer(layer_name):

    float_model = ScalarModel(name=layer_name)

    tpc = mct.get_target_platform_capabilities("6.0")
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(float_model,
                                                                    representative_data_gen=representative_data_gen,
                                                                    target_platform_capabilities=tpc)