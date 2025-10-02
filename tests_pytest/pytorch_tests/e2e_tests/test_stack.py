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
from mct_quantizers import PytorchActivationQuantizationHolder


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


def get_representative_dataset(n_iter=1):

    def representative_dataset() -> Iterator[List]:
        for _ in range(n_iter):
            yield [torch.randn(1, 3, 32, 32)]
    return representative_dataset


def test_stack():

    model = get_model()
    tpc = mct.get_target_platform_capabilities('pytorch', 'imx500') # only imx500 supported
    q_model, _ = mct.ptq.pytorch_post_training_quantization(model, 
                                                            get_representative_dataset(n_iter=1),
                                                            target_resource_utilization=None,
                                                            core_config=mct.core.CoreConfig(),
                                                            target_platform_capabilities=tpc)

    assert hasattr(q_model, 'stack_activation_holder_quantizer') # activation holder for stack layer
    assert isinstance(q_model.stack_activation_holder_quantizer, PytorchActivationQuantizationHolder)
    assert q_model.stack_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8
