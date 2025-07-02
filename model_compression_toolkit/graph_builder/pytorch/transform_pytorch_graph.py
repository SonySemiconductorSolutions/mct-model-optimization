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

from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.residual_collapsing import \
    pytorch_residual_collapsing

from model_compression_toolkit.core.common.substitutions.linear_collapsing_substitution import \
    linear_collapsing_substitute

from model_compression_toolkit.core.pytorch.pytorch_node_prior_info import create_node_prior_info

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.batchnorm_folding import \
    pytorch_batchnorm_folding, pytorch_batchnorm_forward_folding
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.functional_batch_norm import \
    FunctionalBatchNorm
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.functional_layer_norm import \
    FunctionalLayerNorm
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.functional_linear import \
    FunctionalLinear
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.matmul_decomposition import \
    MatMulDecomposition
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.linear_collapsing import \
    pytorch_linear_collapsing
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.multi_head_attention_decomposition \
    import MultiHeadAttentionDecomposition
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.scaled_dot_product_attention import \
    ScaledDotProductDecomposition
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.transform_function_call_method import \
    TransformFunctionCallMethod
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.convtranspose_dynamic_padding import \
    ConvtransposeDynamicPadding
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.const_holder_conv import \
    FunctionalConvSubstitution
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.relu_bound_to_power_of_2 import \
    ReLUBoundToPowerOfTwo
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.remove_identity import RemoveIdentity
from model_compression_toolkit.core.pytorch.graph_substitutions.substitutions.reshape_with_static_shapes import \
    ReshapeWithStaticShapes
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute


def transform_pytorch_graph(graph: Graph,
                            linear_collapsing: bool = True,
                            residual_collapsing: bool = True,
                            relu_bound_to_power_of_2: bool = False) -> Graph:
    """
    Applies PyTorch-specific graph transformations to simplify and optimize the model graph.

    Args:
        graph (Graph): The input computational graph.
        linear_collapsing (bool): If True, collapses consecutive linear operations (e.g., Linear, Conv2D).
        residual_collapsing (bool): If True, simplifies residual connections.
        relu_bound_to_power_of_2 (bool): If True, clips ReLU activation bounds to the nearest upper power-of-2.

    Returns:
        Graph: The transformed computational graph.
    """

    # Normalize graph structure and convert functional forms
    normalization_substitutions = [
        ReshapeWithStaticShapes(),
        MultiHeadAttentionDecomposition(),
        ScaledDotProductDecomposition(),
        MatMulDecomposition(),
        TransformFunctionCallMethod(),
        FunctionalConvSubstitution(),
        FunctionalBatchNorm(),
        FunctionalLayerNorm(),
        FunctionalLinear(),
        RemoveIdentity(),
        ConvtransposeDynamicPadding(),
    ]
    graph = substitute(graph, normalization_substitutions)

    # Annotate nodes with prior information
    for node in graph.nodes:
        node.prior_info = create_node_prior_info(node=node, graph=graph)

    # Apply graph substitutions for folding and quantization prep
    transformation_substitutions = [
        pytorch_batchnorm_folding(),
        pytorch_batchnorm_forward_folding(),
    ]
    if relu_bound_to_power_of_2:
        transformation_substitutions.append(ReLUBoundToPowerOfTwo())

    graph = substitute(graph, transformation_substitutions)

    # Optional linear collapsing
    if linear_collapsing:
        graph = linear_collapsing_substitute(graph, pytorch_linear_collapsing())

    # Optional residual collapsing
    if residual_collapsing:
        graph = substitute(graph, [pytorch_residual_collapsing()])

    return graph

