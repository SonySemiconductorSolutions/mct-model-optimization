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
import pytest
from typing import Iterator, List
import torch
import torch.nn as nn
import model_compression_toolkit as mct
from mct_quantizers import PytorchQuantizationWrapper


class Model(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name

        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.const = nn.Parameter(torch.ones([32, 32]))

    def forward(self, x):
        x = self.conv(x)

        if self.name == 'sin':
            y = torch.sin(self.const) * x
        elif self.name == 'cos':
            y = torch.cos(self.const) * x
        elif self.name == 'exp':
            y = torch.exp(self.const) * x

        return y


def get_representative_dataset(n_iter=1):

    def representative_dataset() -> Iterator[List]:
        for _ in range(n_iter):
            yield [torch.randn(1, 3, 32, 32)]
    return representative_dataset


@pytest.mark.parametrize("tpc_version", [
    '5.0',
    '6.0',
])
@pytest.mark.parametrize("layer", [
    'sin',
    'cos',
    'exp',
])
def test_constant_sin_cos_exp(tpc_version, layer):

    weight_quantizers = []
    
    float_model = Model(layer)
    tpc = mct.get_target_platform_capabilities(tpc_version=tpc_version)
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(float_model, 
                                                                    get_representative_dataset(n_iter=1),
                                                                    target_platform_capabilities=tpc)
    
    weight_quantizers.extend([name for name, module in quantized_model.named_modules() if isinstance(module, PytorchQuantizationWrapper)])

    assert f'{layer}' not in weight_quantizers  # Check that sin, cos, and exp layers do not have the weight quantizer
    assert hasattr(quantized_model, f'{layer}_activation_holder_quantizer')