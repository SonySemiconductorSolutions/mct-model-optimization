# Copyright 2026 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from math import prod
from operator import getitem
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.fx import Graph as FxGraph, GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.functional_node import (
    FunctionalNode,
)
from model_compression_toolkit.core.pytorch.back2framework import (
    pytorch_model_builder,
)
from model_compression_toolkit.core.pytorch.constants import TENSOR_META, TYPE
from model_compression_toolkit.core.pytorch.pytorch_device_config import (
    get_working_device,
)
from model_compression_toolkit.core.pytorch.reader.reader import build_graph
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy


def getitem_graph_factory(
    tensors: dict[str, torch.Tensor],
    index_template: tuple[Any, ...],
    expected: torch.Tensor,
    constants: bool,
) -> tuple[Graph, FunctionalNode]:
    """Build an MCT graph from placeholders or Parameter/buffer references."""
    fx_graph = FxGraph()
    root = nn.Module()
    references = {}
    for name, tensor in tensors.items():
        if constants:
            if name == "bias":
                root.register_parameter(name, nn.Parameter(tensor))
            else:
                root.register_buffer(name, tensor)
            reference = fx_graph.get_attr(name)
        else:
            reference = fx_graph.placeholder(name)
        reference.meta = {
            TYPE: nn.Parameter if constants and name == "bias"
            else torch.Tensor,
            TENSOR_META: _extract_tensor_metadata(tensor),
        }
        references[name] = reference
    index = tuple(
        references[value] if isinstance(value, str) else value
        for value in index_template
    )
    source = references[next(iter(tensors))]
    getitem_node = fx_graph.call_function(getitem, (source, index))
    getitem_node.meta = {
        TYPE: torch.Tensor,
        TENSOR_META: _extract_tensor_metadata(expected),
    }
    fx_graph.output(getitem_node)
    graph_module = GraphModule(root, fx_graph)
    graph_module.graph.lint()
    graph = build_graph(graph_module, torch_tensor_to_numpy)
    nodes = [
        node for node in graph.nodes
        if isinstance(node, FunctionalNode)
        and node.functional_op is getitem
    ]
    assert len(nodes) == 1
    node = nodes[0]
    assert graph.get_outputs()[0].node is node
    return graph, node


def assert_getitem_execution(
    graph: Graph,
    node: FunctionalNode,
    tensors: dict[str, torch.Tensor],
    index_template: tuple[Any, ...],
    expected: torch.Tensor,
) -> None:
    """Collect real edge inputs and inspect arguments at the getitem boundary."""
    inputs = tuple(
        tensors[input_node.name] for input_node in graph.get_inputs()
    )
    node_outputs = {
        input_node: pytorch_model_builder._build_input_tensors_list(
            input_node, graph, inputs, {}
        )
        for input_node in graph.get_inputs()
    }
    input_tensors = pytorch_model_builder._build_input_tensors_list(
        node, graph, inputs, node_outputs
    )
    edges = graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX)
    assert len(input_tensors) == len(edges)
    for tensor, edge in zip(input_tensors, edges):
        assert tensor is tensors[edge.source_node.name]
    source_name = next(iter(tensors))
    calls = []

    def checked_getitem(
        source: torch.Tensor, index: tuple[Any, ...]
    ) -> torch.Tensor:
        """Check restored inputs and delegate to the real getitem operation."""
        calls.append((source, index))
        assert source.dtype == torch.float32
        torch.testing.assert_close(
            source, tensors[source_name], rtol=0, atol=0
        )
        if not node.weights:
            assert source is tensors[source_name]
        assert isinstance(index, tuple)
        assert len(index) == len(index_template)
        for actual, template in zip(index, index_template):
            if isinstance(template, str):
                assert isinstance(actual, torch.Tensor)
                assert actual.dtype == torch.long
                torch.testing.assert_close(
                    actual, tensors[template], rtol=0, atol=0
                )
                if not node.weights:
                    assert actual is tensors[template]
            else:
                assert actual == template
        return getitem(source, index)

    outputs, float_outputs = pytorch_model_builder._run_operation(
        node, input_tensors, checked_getitem,
        quantize_node_activation_fn=None,
        use_activation_quantization=False,
    )
    assert len(calls) == 1
    assert len(outputs) == len(float_outputs) == 1
    assert outputs[0] is float_outputs[0]
    assert tuple(outputs[0].shape) == tuple(expected.shape)
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape, index_template, index_values, allocations, saved_index,"
    " expected_values",
    [
        pytest.param(
            (2, 3), (slice(None), "cols"), {"cols": [2, 0]},
            [0, (1, 1)], (slice(None), None),
            [[2, 0], [5, 3]], id="x[:, cols]",
        ),
        pytest.param(
            (2, 3), ("rows", slice(None)), {"rows": [1, 0]},
            [0, (1, 0)], (None, slice(None)),
            [[3, 4, 5], [0, 1, 2]], id="x[rows, :]",
        ),
        pytest.param(
            (2, 3, 4), (slice(None), "cols", slice(None)),
            {"cols": [2, 0]},
            [0, (1, 1)], (slice(None), None, slice(None)),
            [[[8, 9, 10, 11], [0, 1, 2, 3]],
             [[20, 21, 22, 23], [12, 13, 14, 15]]],
            id="x[:, cols, :]",
        ),
        pytest.param(
            (2, 3, 4), (Ellipsis, "cols"), {"cols": [2, 0]},
            [0, (1, 1)], (Ellipsis, None),
            [[[2, 0], [6, 4], [10, 8]],
             [[14, 12], [18, 16], [22, 20]]], id="x[..., cols]",
        ),
        pytest.param(
            (2, 3), ("rows", "cols"),
            {"rows": [1, 0], "cols": [2, 0]},
            [0, (1, 0), (1, 1)], (None, None), [5, 0],
            id="x[rows, cols]",
        ),
        pytest.param(
            (2, 3), ("rows",), {"rows": [1, 0]},
            [0, (1, 0)], (None,), [[3, 4, 5], [0, 1, 2]],
            id="x[rows,]",
        ),
        pytest.param(
            (2, 3), (slice(None), 0), {},
            [0], (slice(None), 0), [0, 3], id="x[:, 0]",
        ),
        pytest.param(
            (2, 3), (None, slice(None), "cols"), {"cols": [2, 0]},
            [0, (1, 2)], (None, slice(None), None),
            [[[2, 0], [5, 3]]], id="x[None, :, cols]",
        ),
    ],
)
def test_getitem_tuple_indices(
    shape: tuple[int, ...],
    index_template: tuple[Any, ...],
    index_values: dict[str, list[int]],
    allocations: list[int | tuple[int, int]],
    saved_index: tuple[Any, ...],
    expected_values: list[Any],
) -> None:
    """Integrate reader graph generation with builder input restoration."""
    device = get_working_device()
    source = torch.arange(
        prod(shape), dtype=torch.float32, device=device
    ).reshape(shape)
    tensors = {
        "x": source,
        **{
            name: torch.tensor(values, dtype=torch.long, device=device)
            for name, values in index_values.items()
        },
    }
    expected = torch.tensor(
        expected_values, dtype=torch.float32, device=device
    )
    graph, node = getitem_graph_factory(
        tensors, index_template, expected, False
    )
    edges = graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX)
    assert [edge.sink_index for edge in edges] == list(range(len(tensors)))
    assert [edge.source_index for edge in edges] == [0] * len(tensors)
    assert [edge.source_node for edge in edges] == graph.get_inputs()
    assert [
        input_node.name for input_node in graph.get_inputs()
    ] == list(tensors)
    assert node.weights == {}
    assert node.tensor_input_allocs == allocations
    assert node.op_call_args == [saved_index]
    original_args = node.op_call_args
    original_index = node.op_call_args[0]
    original_allocations = node.tensor_input_allocs
    for offset in (0, 1):
        runtime_tensors = {
            "x": source + offset,
            **{name: tensors[name].clone() for name in index_values},
        }
        assert_getitem_execution(
            graph, node, runtime_tensors, index_template, expected + offset
        )
        assert node.op_call_args is original_args
        assert node.op_call_args == [saved_index]
        assert node.op_call_args[0] is original_index
        assert node.tensor_input_allocs is original_allocations
        assert node.tensor_input_allocs == allocations
        assert node.op_call_kwargs == {}


@pytest.mark.parametrize(
    "index_template, index_values, allocations, saved_index, expected_values",
    [
        pytest.param(
            (slice(None), "indices"), {"indices": [2, 0]},
            [0, (1, 1)], (slice(None), None),
            [[2, 0], [5, 3]], id="bias[:, indices]",
        ),
        pytest.param(
            ("indices", slice(None)), {"indices": [1, 0]},
            [0, (1, 0)], (None, slice(None)),
            [[3, 4, 5], [0, 1, 2]], id="bias[indices, :]",
        ),
        pytest.param(
            (slice(None), "indices"), {"indices": [[2, 0], [1, 2]]},
            [0, (1, 1)], (slice(None), None),
            [[[2, 0], [1, 2]], [[5, 3], [4, 5]]],
            id="bias[:, indices] (matrix)",
        ),
        pytest.param(
            ("rows", "cols"), {"rows": [1, 0], "cols": [2, 0]},
            [0, (1, 0), (1, 1)], (None, None),
            [5, 0], id="bias[rows, cols]",
        ),
    ],
)
def test_getitem_constant_tuple_indices(
    index_template: tuple[Any, ...],
    index_values: dict[str, list[int] | list[list[int]]],
    allocations: list[int | tuple[int, int]],
    saved_index: tuple[Any, ...],
    expected_values: list[Any],
) -> None:
    """Integrate constant extraction, insertion, conversion and restoration."""
    device = get_working_device()
    tensors = {
        "bias": torch.arange(
            6, dtype=torch.float32, device=device
        ).reshape(2, 3),
        **{
            name: torch.tensor(values, dtype=torch.long, device=device)
            for name, values in index_values.items()
        },
    }
    expected = torch.tensor(
        expected_values, dtype=torch.float32, device=device
    )
    graph, node = getitem_graph_factory(
        tensors, index_template, expected, True
    )
    assert graph.get_inputs() == []
    assert list(graph.incoming_edges(node)) == []
    assert set(node.weights) == set(range(len(tensors)))
    assert node.tensor_input_allocs == allocations
    assert node.op_call_args == [saved_index]
    original_args = node.op_call_args
    original_index = node.op_call_args[0]
    original_allocations = node.tensor_input_allocs
    references = [
        torch_tensor_to_numpy(tensor).copy() for tensor in tensors.values()
    ]
    for _ in range(2):
        assert_getitem_execution(
            graph, node, tensors, index_template, expected
        )
        assert node.op_call_args is original_args
        assert node.op_call_args == [saved_index]
        assert node.op_call_args[0] is original_index
        assert node.tensor_input_allocs is original_allocations
        assert node.tensor_input_allocs == allocations
        assert node.op_call_kwargs == {}
        for position, reference in enumerate(references):
            saved_constant = node.weights[position]
            assert isinstance(saved_constant, np.ndarray)
            assert saved_constant.shape == reference.shape
            assert saved_constant.dtype == reference.dtype
            np.testing.assert_array_equal(saved_constant, reference)
