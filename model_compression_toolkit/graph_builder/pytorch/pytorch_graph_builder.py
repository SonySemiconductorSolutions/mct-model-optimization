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

import torch.nn

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.graph_builder.pytorch.convert_pytorch_model_to_graph import \
    convert_pytorch_model_to_graph
from model_compression_toolkit.graph_builder.common.base_graph_builder import BaseGraphBuilder
from model_compression_toolkit.graph_builder.pytorch.transform_pytorch_graph import transform_pytorch_graph


class PytorchGraphBuilder(BaseGraphBuilder):
    """
    Graph builder for PyTorch models.

    This class handles:
    1. Converting a `torch.nn.Module` into a framework-agnostic computational graph.
    2. Applying PyTorch-specific graph transformations that are needed for pytorch models such as ReshapeWithStaticShapes,
       and optional transformations such as linear collapsing, residual simplification, and ReLU clipping adjustments.
    """

    def convert_model_to_graph(self, model: torch.nn.Module, representative_dataset: Callable = None) -> Graph:
        """
        Converts a PyTorch model to a computational graph.

        Args:
            model (torch.nn.Module): The PyTorch model to convert.
            representative_dataset (Callable): A callable that yields representative input samples
                                               (required for tracing the model graph).

        Returns:
            Graph: A graph representation of the given PyTorch model.

        Raises:
            ValueError: If no representative dataset is provided.
        """
        if representative_dataset is None:
            raise ValueError("PyTorch requires a representative_dataset to convert the model.")
        return convert_pytorch_model_to_graph(model, representative_dataset)

    def transform_graph(self,
                        graph: Graph,
                        linear_collapsing: bool = True,
                        residual_collapsing: bool = True,
                        relu_bound_to_power_of_2: bool = False) -> Graph:
        """
        Applies PyTorch-specific graph transformations to simplify the graph.

        Args:
            graph (Graph): The graph generated from a PyTorch model.
            linear_collapsing (bool): If True, collapses consecutive linear layers (e.g., Linear, Conv2d).
            residual_collapsing (bool): If True, simplifies residual (skip) connections into linear operations.
            relu_bound_to_power_of_2 (bool): If True, clips ReLU activation bounds to the upper power-of-2 value.

        Returns:
            Graph: The transformed, optimized graph.
        """
        return transform_pytorch_graph(graph,
                                       linear_collapsing,
                                       residual_collapsing,
                                       relu_bound_to_power_of_2)
