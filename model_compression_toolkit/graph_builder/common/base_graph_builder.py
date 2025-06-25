from abc import ABC, abstractmethod
from typing import Any

from model_compression_toolkit.core.common import Graph


class BaseGraphBuilder(ABC):

    def build_graph(self,
                    model,
                    representative_dataset=None,
                    fqc=None,
                    tensorboard_writer=None,
                    linear_collapsing: bool = True,
                    residual_collapsing: bool = True,
                    relu_bound_to_power_of_2: bool = False):
        graph = self.convert_model_to_graph(model, representative_dataset)
        if tensorboard_writer is not None:
            tensorboard_writer.add_graph(graph, 'initial_graph')

        # TODO: Temporary until it can be removed from the graph
        # Some tests do not use the fqc, so we set only if it is passed.
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
    def convert_model_to_graph(self, model: Any, representative_dataset: Any = None) -> Graph:
        raise Exception

    @abstractmethod
    def transform_graph(self,
                        graph: Graph,
                        linear_collapsing: bool = True,
                        residual_collapsing: bool = True,
                        relu_bound_to_power_of_2: bool = False) -> Graph:
        raise Exception
