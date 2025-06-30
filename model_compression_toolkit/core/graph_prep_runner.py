# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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


from typing import Callable, Any

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_nodes_candidates
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.common.substitutions.linear_collapsing_substitution import \
    linear_collapsing_substitute
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities


# TODO: whoever calls graph_preparation_runner should now call get_finalized_graph directly after
#  going through the graph builder module

def get_finalized_graph(graph: Graph,
                        fqc: FrameworkQuantizationCapabilities,
                        quant_config: QuantizationConfig = DEFAULTCONFIG,
                        bit_width_config: BitWidthConfig = None,
                        tb_w: TensorboardWriter = None,
                        fw_impl: FrameworkImplementation = None,
                        mixed_precision_enable: bool = False,
                        running_gptq: bool = False) -> Graph:
    """
    Applies all edit operation (edit, substitutions, etc.) on the model's graph, to prepare it for the quantization
    process. All future graph substitutions and operations that change the graph should be added to this method.

    Args:
        initial_graph (Graph): Graph to apply the changes to.
        fqc (FrameworkQuantizationCapabilities): FrameworkQuantizationCapabilities object that describes the desired inference target platform (includes fusing patterns MCT should handle).
        quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be
            quantized.
        bit_width_config (BitWidthConfig): Config for bit-width selection. Defaults to None.
        tb_w (TensorboardWriter): TensorboardWriter object to use for logging events such as graphs, histograms, etc.
        fw_impl (FrameworkImplementation): FrameworkImplementation object with a specific framework methods implementation.
        mixed_precision_enable: is mixed precision enabled.
        running_gptq: Whether or not a GPTQ optimization is planned to run after the PTQ process.

    Returns: Graph object that represents the model, after applying all required modifications to it.
    """

    ######################################
    # Add quantization configurations
    ######################################
    transformed_graph = set_quantization_configuration_to_graph(graph=graph,
                                                                quant_config=quant_config,
                                                                bit_width_config=bit_width_config,
                                                                mixed_precision_enable=mixed_precision_enable,
                                                                running_gptq=running_gptq)

    ######################################
    # Layer fusing
    ######################################
    fusing_info = FusingInfoGenerator(fqc.get_fusing_patterns()).generate_fusing_info(transformed_graph)
    transformed_graph.fusing_info = fusing_info
    transformed_graph.disable_fused_nodes_activation_quantization()

    ######################################
    # Channel equalization
    ######################################
    transformed_graph = substitute(transformed_graph,
                                   fw_impl.get_substitutions_channel_equalization(quant_config))

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_graph_marking')

    ######################################
    # Filter nodes' candidates
    ######################################
    transformed_graph = filter_nodes_candidates(transformed_graph)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_candidates_filtering')

    return transformed_graph


