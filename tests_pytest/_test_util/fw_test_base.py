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
import abc
from typing import Callable

from model_compression_toolkit.core.graph_prep_runner import get_finalized_graph

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.graph_builder.common.base_graph_builder import BaseGraphBuilder


class BaseFWIntegrationTest(abc.ABC):
    """ Base class providing utils for integration / e2e tests. """

    fw_impl: FrameworkImplementation
    attach_to_fw_func: Callable
    fw_graph_builder: BaseGraphBuilder

    def run_graph_preparation(self, model, datagen, tpc, quant_config=None,
                              mp: bool = False, gptq: bool = False, bit_width_config=None):
        quant_config = quant_config or QuantizationConfig()
        fqc = self.attach_to_fw_func(tpc)
        graph = self.fw_graph_builder.build_graph(model=model,
                                                  representative_dataset=datagen,
                                                  fqc=fqc,
                                                  linear_collapsing=quant_config.linear_collapsing,
                                                  residual_collapsing=quant_config.residual_collapsing,
                                                  relu_bound_to_power_of_2=quant_config.relu_bound_to_power_of_2)

        graph = get_finalized_graph(graph=graph,
                                    fqc=fqc,
                                    quant_config=quant_config,
                                    bit_width_config=bit_width_config,
                                    fw_impl=self.fw_impl,
                                    mixed_precision_enable=mp,
                                    running_gptq=gptq)

        return graph
