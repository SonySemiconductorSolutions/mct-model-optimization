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

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter

from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities

from model_compression_toolkit.core.common import Graph


class BaseGraphBuilder(ABC):
    """
    Abstract base class for converting framework-specific models into a framework-agnostic computational graph,
    and applying standard graph transformations.

    Subclasses must implement methods to:
    1. Convert a model to a graph representation.
    2. Apply optional graph simplifications and preprocessing.
    """

    def build_graph(self,
                    model: Any,
                    representative_dataset: Callable = None,
                    fqc: FrameworkQuantizationCapabilities = None,
                    tensorboard_writer: TensorboardWriter = None,
                    linear_collapsing: bool = True,
                    residual_collapsing: bool = True,
                    relu_bound_to_power_of_2: bool = False):
        """
        Converts a model to a graph and applies selected graph transformations. Optionally logs the graph
        before and after transformation using TensorBoard.

        Args:
            model (Any): The framework-specific model to convert.
            representative_dataset (Callable, optional): A callable that yields representative inputs for tracing
                (used in some frameworks like PyTorch).
            fqc (FrameworkQuantizationCapabilities, optional): Object containing quantization capabilities of the framework.
                If provided, it's stored in the graph.
            tensorboard_writer (TensorboardWriter, optional): If provided, logs the initial and transformed graphs.
            linear_collapsing (bool): Whether to collapse sequences of linear layers (e.g., dense or conv).
            residual_collapsing (bool): Whether to collapse residual connections.
            relu_bound_to_power_of_2 (bool): If True, replaces ReLU activation bounds with nearest power-of-2 (for HW-friendliness).

        Returns:
            Graph: The transformed, simplified computational graph.
        """
        graph = self.convert_model_to_graph(model, representative_dataset)
        if tensorboard_writer is not None:
            tensorboard_writer.add_graph(graph, 'initial_graph')

        if fqc:
            graph.set_fqc(fqc)

        transformed_graph = self.transform_graph(graph,
                                                 linear_collapsing,
                                                 residual_collapsing,
                                                 relu_bound_to_power_of_2)
        if tensorboard_writer is not None:
            tensorboard_writer.add_graph(transformed_graph, 'after_graph_preparation')

        return transformed_graph

    @abstractmethod
    def convert_model_to_graph(self, model: Any, representative_dataset: Optional[Callable] = None) -> Graph:
        """
        Converts a framework-specific model to a framework-agnostic graph representation.

        Args:
            model (Any): The model to convert.
            representative_dataset (Callable, optional): A dataset generator or iterator, if required for tracing like in Pytorch.

        Returns:
            Graph: A computational graph representation of the model.
        """
        raise ValueError("Implementation for convert_model_to_graph is missing")  # pragma: no cover

    @abstractmethod
    def transform_graph(self,
                        graph: Graph,
                        linear_collapsing: bool = True,
                        residual_collapsing: bool = True,
                        relu_bound_to_power_of_2: bool = False) -> Graph:
        """
        Applies basic transformations and optimizations to the graph.

        Args:
            graph (Graph): The input computational graph.
            linear_collapsing (bool): Whether to merge consecutive linear operations.
            residual_collapsing (bool): Whether to simplify residual connections into linear nodes.
            relu_bound_to_power_of_2 (bool): Whether to adapt ReLU clipping bounds to power-of-2 values.

        Returns:
            Graph: The transformed computational graph.
        """
        raise ValueError("Implementation for transform_graph is missing")  # pragma: no cover

