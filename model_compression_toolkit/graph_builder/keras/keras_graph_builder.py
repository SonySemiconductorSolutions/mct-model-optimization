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

from typing import Any, Callable

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.graph_builder.keras.convert_keras_model_to_graph import convert_keras_model_to_graph
from model_compression_toolkit.graph_builder.common.base_graph_builder import BaseGraphBuilder
from model_compression_toolkit.graph_builder.keras.transform_keras_graph import transform_keras_graph


class KerasGraphBuilder(BaseGraphBuilder):
    """
    Graph builder for Keras models.

    This class handles:
    1. Converting a `tf.keras.Model` into a framework-agnostic computational graph.
    2. Applying Keras-specific graph transformations such as separable and activation decomposition,
       along with other optional modifications controlled by configuration flags.
    """

    def convert_model_to_graph(self, model: Any, representative_dataset: Callable = None) -> Graph:
        """
        Converts a Keras model to a computational graph.

        Args:
            model: A Keras model instance.
            representative_dataset: Not used for Keras, included for API compatibility.

        Returns:
            Graph: A graph representation of the given Keras model.
        """
        return convert_keras_model_to_graph(model)

    def transform_graph(self,
                        graph: Graph,
                        linear_collapsing: bool = True,
                        residual_collapsing: bool = True,
                        relu_bound_to_power_of_2: bool = False) -> Graph:
        """
        Applies Keras-specific graph transformations to simplify and optimize the graph.

        Args:
            graph (Graph): The graph generated from a Keras model.
            linear_collapsing (bool): If True, collapses consecutive linear layers (e.g., Dense, Conv2D).
            residual_collapsing (bool): If True, simplifies residual connections into linear operations.
            relu_bound_to_power_of_2 (bool): If True, adjusts ReLU activation clipping bounds to the upper power-of-2 bound.

        Returns:
            Graph: The transformed graph.
        """
        return transform_keras_graph(graph,
                                     linear_collapsing,
                                     residual_collapsing,
                                     relu_bound_to_power_of_2)

