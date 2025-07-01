from copy import copy
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.graph_builder.keras.reader.common import is_node_a_model
from model_compression_toolkit.graph_builder.keras.reader.reader import build_connectivity_handler, build_graph, flatten_nested_model
import tensorflow as tf

def convert_keras_model_to_graph(model: tf.keras.Model) -> Graph:
    """
    Converts a Keras model into a computational graph.

    This function analyzes the structure of a Keras model
    and builds a graph representation from its layers and connections.

    Args:
        model (tf.keras.Model): The Keras model to convert.

    Returns:
        ComputationalGraph: A graph containing nodes and edges representing the model.
    """
    connectivity_handler = build_connectivity_handler(model)
    model_graph = build_graph(model, connectivity_handler)

    # Go over all nodes in the graph, and if one of them is a model by itself, unroll it recursively by
    # merging the inner model's graph into the outer model's graph.
    nodes = copy(model_graph.nodes)
    for node in nodes:
        if is_node_a_model(node):  # if the node represents a Keras model - flat it recursively
            model_graph = flatten_nested_model(model_graph,
                                               node,
                                               model)
    return model_graph
