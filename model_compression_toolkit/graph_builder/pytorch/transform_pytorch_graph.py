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
    Applies a series of structural simplifications to a graph.

    This includes transformations such as batch normalization folding, merging linear layers, etc.
    These transformations are aimed at simplifying the graph for optimization without altering the model's
    functionality.

    Args:
        graph (Graph): The input graph to transform.
        linear_collapsing:
        residual_collapsing:
        relu_bound_to_power_of_2:

    Returns:
        Graph: A refined graph with structural transformations applied.

    Notes:
        This function does not perform numerical optimizations (e.g., quantization),
        nor does it alter weights or model accuracy. It is purely structural.
    """
    prepare_graph_substitutions = [ReshapeWithStaticShapes(),
                                   MultiHeadAttentionDecomposition(),
                                   ScaledDotProductDecomposition(),
                                   MatMulDecomposition(),
                                   TransformFunctionCallMethod(),
                                   FunctionalConvSubstitution(),
                                   FunctionalBatchNorm(),
                                   FunctionalLayerNorm(),
                                   FunctionalLinear(),
                                   RemoveIdentity(),
                                   ConvtransposeDynamicPadding()]
    graph = substitute(graph, prepare_graph_substitutions)

    # **************************************************

    for node in graph.nodes:
        node.prior_info = create_node_prior_info(node=node, graph=graph)

    # **************************************************
    substitutions_list = [pytorch_batchnorm_folding(),
                          pytorch_batchnorm_forward_folding()]
    if relu_bound_to_power_of_2:
        substitutions_list.append(ReLUBoundToPowerOfTwo())

    graph = substitute(graph, substitutions_list)
    # **************************************************

    if linear_collapsing:
        graph = linear_collapsing_substitute(graph, pytorch_linear_collapsing())

    if residual_collapsing:
        graph = substitute(graph, [pytorch_residual_collapsing()])

    return graph
