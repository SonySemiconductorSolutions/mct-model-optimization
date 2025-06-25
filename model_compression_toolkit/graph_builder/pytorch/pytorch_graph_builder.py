from typing import Any

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.graph_builder import convert_pytorch_model_to_graph
from model_compression_toolkit.graph_builder.common.base_graph_builder import BaseGraphBuilder
from model_compression_toolkit.graph_builder.pytorch.transform_pytorch_graph import transform_pytorch_graph


class PytorchGraphBuilder(BaseGraphBuilder):
    def convert_model_to_graph(self, model: Any, representative_dataset: Any = None) -> Graph:
        if representative_dataset is None:
            raise ValueError("PyTorch requires a representative_dataset to convert the model.")
        return convert_pytorch_model_to_graph(model, representative_dataset)

    def transform_graph(self,
                        graph: Graph,
                        linear_collapsing: bool = True,
                        residual_collapsing: bool = True,
                        relu_bound_to_power_of_2: bool = False
                        ) -> Graph:
        return transform_pytorch_graph(graph,
                                       linear_collapsing,
                                       residual_collapsing,
                                       relu_bound_to_power_of_2)
