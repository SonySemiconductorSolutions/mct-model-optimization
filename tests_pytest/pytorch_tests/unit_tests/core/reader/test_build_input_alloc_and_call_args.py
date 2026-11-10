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
from operator import getitem
from typing import Any

import pytest
from torch.fx import Graph

from model_compression_toolkit.core.pytorch.reader.graph_builders import (
    _build_input_alloc_and_call_args,
)


class TestBuildGetitemInputAllocAndCallArgs:
    """Verify allocations and saved arguments for operator.getitem index tuples."""

    @pytest.mark.parametrize(
        "index_template, expected_index, expected_allocations",
        [
            pytest.param((slice(None), "cols"), (slice(None), None),
                         [0, (1, 1)], id="column"),
            pytest.param(("rows", slice(None)), (None, slice(None)),
                         [0, (1, 0)], id="row"),
            pytest.param((slice(None), "cols", slice(None)),
                         (slice(None), None, slice(None)),
                         [0, (1, 1)], id="middle"),
            pytest.param((Ellipsis, "cols"), (Ellipsis, None),
                         [0, (1, 1)], id="ellipsis"),
            pytest.param(("rows", "cols"), (None, None),
                         [0, (1, 0), (1, 1)], id="pair"),
            pytest.param(("rows",), (None,), [0, (1, 0)], id="singleton"),
            pytest.param((slice(None), 0), (slice(None), 0),
                         [0], id="fixed_tuple"),
            pytest.param((None, slice(None), "cols"),
                         (None, slice(None), None),
                         [0, (1, 2)], id="newaxis"),
        ],
    )
    def test_getitem_index_allocations(
        self,
        index_template: tuple[Any, ...],
        expected_index: tuple[Any, ...],
        expected_allocations: list[int | tuple[int, int]],
    ) -> None:
        """Check allocations and placeholders for operator.getitem index tuples."""
        graph = Graph()
        x = graph.placeholder("x")
        references = {
            "rows": graph.placeholder("rows"),
            "cols": graph.placeholder("cols"),
        }
        index = tuple(
            references[value] if isinstance(value, str) else value
            for value in index_template
        )
        node = graph.call_function(getitem, (x, index))

        call_args, allocations = _build_input_alloc_and_call_args(
            node, {}, False
        )

        assert allocations == expected_allocations
        assert call_args == [x, expected_index]

    def test_getitem_original_args_unchanged(self) -> None:
        """Keep operator.getitem FX references unchanged across two reads."""
        graph = Graph()
        x = graph.placeholder("x")
        rows = graph.placeholder("rows")
        cols = graph.placeholder("cols")
        node = graph.call_function(getitem, (x, (rows, cols)))
        original_args = node.args
        original_index = node.args[1]

        for _ in range(2):
            call_args, allocations = _build_input_alloc_and_call_args(
                node, {}, False
            )
            assert allocations == [0, (1, 0), (1, 1)]
            assert call_args == [x, (None, None)]
            assert node.args is original_args
            assert node.args[1] is original_index
            assert node.args[0] is x
            assert node.args[1][0] is rows
            assert node.args[1][1] is cols
