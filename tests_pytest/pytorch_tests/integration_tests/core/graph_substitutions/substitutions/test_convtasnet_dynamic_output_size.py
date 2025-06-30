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
import torch
from model_compression_toolkit.core.graph_prep_runner import get_finalized_graph
from torch import nn

from model_compression_toolkit.graph_builder.pytorch.pytorch_graph_builder import PytorchGraphBuilder
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.pytorch.default_framework_info import PyTorchInfo
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.upsample(x, output_size=[224, 224])  # <--- dynamic output_size
        x = self.downsample2(x)
        x = self.upsample2(x)  # <--- no dynamic output_size
        return x


def data_gen():
    yield [torch.rand(1, 3, 224, 224)]


def test_convtranspose_dynamic_output_size(minimal_tpc):
    Model()(next(data_gen())[0])

    set_fw_info(PyTorchInfo)
    fw_impl = PytorchImplementation()
    fw_graph_builder = PytorchGraphBuilder()
    model = Model()
    fqc = AttachTpcToPytorch().attach(minimal_tpc)
    quant_config = QuantizationConfig()

    graph = fw_graph_builder.build_graph(model=model,
                                         representative_dataset=data_gen,
                                         fqc=fqc,
                                         linear_collapsing=quant_config.linear_collapsing,
                                         residual_collapsing=quant_config.residual_collapsing,
                                         relu_bound_to_power_of_2=quant_config.relu_bound_to_power_of_2)

    graph = get_finalized_graph(graph=graph,
                                fqc=fqc,
                                quant_config=quant_config,
                                fw_impl=fw_impl,
                                mixed_precision_enable=False,
                                running_gptq=False)

    nodes = graph.get_topo_sorted_nodes()

    assert nodes[2].framework_attr['output_padding'] == (1,1)
    assert nodes[2].output_shape[0] == [1, 3, 224, 224]
    assert nodes[4].framework_attr['output_padding'] == (0,0)
    assert nodes[4].output_shape[0] == [1, 3, 223, 223]
