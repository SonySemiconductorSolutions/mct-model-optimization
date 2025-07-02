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

from model_compression_toolkit.core.common.substitutions.linear_collapsing_substitution import linear_collapsing_substitute
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.remove_identity import RemoveIdentity
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.activation_decomposition import ActivationDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.matmul_substitution import MatmulToDenseSubstitution
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.sigmoid_mul_to_swish import MulSigmoidToSwish
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.conv_funcs_to_layer import Conv2dFuncToConv2dLayer, DwConv2dFuncToDwConv2dLayer
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.linear_collapsing import keras_linear_collapsing, keras_op2d_add_const_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.residual_collapsing import keras_residual_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.relu_bound_to_power_of_2 import ReLUBoundToPowerOfTwo
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.multi_head_attention_decomposition import MultiHeadAttentionDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.separableconv_decomposition import SeparableConvDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.dwconv_to_conv import DwconvToConv
from model_compression_toolkit.core.keras.keras_node_prior_info import create_node_prior_info
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_folding import keras_batchnorm_folding, keras_batchnorm_forward_folding


def transform_keras_graph(graph: Graph,
                          linear_collapsing: bool = True,
                          residual_collapsing: bool = True,
                          relu_bound_to_power_of_2: bool = False) -> Graph:
    """
    Applies a series of Keras-specific graph transformations to simplify the model graph.

    Args:
        graph (Graph): The graph to transform.
        linear_collapsing (bool): If True, applies transformations that collapse consecutive linear ops.
        residual_collapsing (bool): If True, collapses residual connections into simplified nodes.
        relu_bound_to_power_of_2 (bool): If True, clips ReLU bounds to the upper power of 2 bound.

    Returns:
        Graph: The transformed graph.
    """

    # Initial substitutions to normalize and standardize the graph structure
    normalization_substitutions = [
        MulSigmoidToSwish(),
        SeparableConvDecomposition(),
        MatmulToDenseSubstitution(),
        Conv2dFuncToConv2dLayer(),
        DwConv2dFuncToDwConv2dLayer(),
        MultiHeadAttentionDecomposition(),
        ActivationDecomposition(),
        DwconvToConv(),
        RemoveIdentity(),
    ]
    graph = substitute(graph, normalization_substitutions)

    # Populate prior information needed for further processing
    for node in graph.nodes:
        node.prior_info = create_node_prior_info(node=node, graph=graph)

    # BatchNorm-related transformations
    transformation_substitutions = [
        keras_batchnorm_folding(),
        keras_batchnorm_forward_folding()
    ]

    if relu_bound_to_power_of_2:
        transformation_substitutions.append(ReLUBoundToPowerOfTwo())

    graph = substitute(graph, transformation_substitutions)

    # Linear collapsing transformations
    if linear_collapsing:
        graph = linear_collapsing_substitute(graph, keras_linear_collapsing())
        graph = linear_collapsing_substitute(graph, keras_op2d_add_const_collapsing())

    # Residual connection collapsing
    if residual_collapsing:
        graph = substitute(graph, [keras_residual_collapsing()])

    return graph
