import copy
from typing import Callable
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy, to_torch_tensor
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.graph_builder.pytorch.reader.reader import model_reader
import torch

def convert_pytorch_model_to_graph(model: torch.nn.Module,
                                   representative_dataset: Callable) -> Graph:
    """
    Converts a PyTorch model into a computational graph using tracing.

    This function requires a representative dataset to trace the dynamic
    execution of the model and build a corresponding graph.

    Args:
        model: The PyTorch model to convert.
        representative_dataset: An iterable yielding sample inputs for tracing the model.

    Returns:
        Graph: A graph containing nodes and edges representing the model.

    """
    _module = copy.deepcopy(model)
    _module.eval()
    return model_reader(_module,
                        representative_dataset,
                        torch_tensor_to_numpy,
                        to_torch_tensor)
