from model_compression_toolkit.core.common.substitutions.linear_collapsing_substitution import \
    linear_collapsing_substitute

from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute

from model_compression_toolkit.core.keras.graph_substitutions.substitutions.remove_identity import RemoveIdentity
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.activation_decomposition import \
    ActivationDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.matmul_substitution import \
    MatmulToDenseSubstitution
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.sigmoid_mul_to_swish import \
    MulSigmoidToSwish
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.conv_funcs_to_layer import \
    Conv2dFuncToConv2dLayer, DwConv2dFuncToDwConv2dLayer
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.linear_collapsing import \
    keras_linear_collapsing, keras_op2d_add_const_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.residual_collapsing import \
    keras_residual_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.relu_bound_to_power_of_2 import \
    ReLUBoundToPowerOfTwo
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.multi_head_attention_decomposition import \
    MultiHeadAttentionDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.separableconv_decomposition import \
    SeparableConvDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.dwconv_to_conv import DwconvToConv
from model_compression_toolkit.core.keras.keras_node_prior_info import create_node_prior_info
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_folding import \
    keras_batchnorm_folding, keras_batchnorm_forward_folding


def transform_keras_graph(graph: Graph,
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
    prepare_graph_substitutions = [MulSigmoidToSwish(),
                                   SeparableConvDecomposition(),
                                   MatmulToDenseSubstitution(),
                                   Conv2dFuncToConv2dLayer(),
                                   DwConv2dFuncToDwConv2dLayer(),
                                   MultiHeadAttentionDecomposition(),
                                   ActivationDecomposition(),
                                   DwconvToConv(),
                                   RemoveIdentity()]

    graph = substitute(graph,
                       prepare_graph_substitutions)

    # **************************************************

    for node in graph.nodes:
        node.prior_info = create_node_prior_info(node=node, graph=graph)

    # **************************************************

    substitutions_list = [keras_batchnorm_folding(),
                          keras_batchnorm_forward_folding()]
    if relu_bound_to_power_of_2:
        substitutions_list.append(ReLUBoundToPowerOfTwo())

    graph = substitute(graph, substitutions_list)
    # **************************************************

    if linear_collapsing:
        graph = linear_collapsing_substitute(graph, keras_linear_collapsing())
        graph = linear_collapsing_substitute(graph, keras_op2d_add_const_collapsing())

    if residual_collapsing:
        graph = substitute(graph, [keras_residual_collapsing()])

    return graph
