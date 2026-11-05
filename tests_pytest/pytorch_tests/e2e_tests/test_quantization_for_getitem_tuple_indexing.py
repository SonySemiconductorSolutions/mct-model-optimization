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
from collections.abc import Iterator
from functools import partial
from math import prod
from operator import add, getitem, mul

import numpy as np
import pytest
import torch
from mct_quantizers import PytorchActivationQuantizationHolder
from torch import nn

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.graph.functional_node import (
    FunctionalNode,
)
from model_compression_toolkit.core.pytorch.back2framework import (
    pytorch_model_builder,
)
from model_compression_toolkit.core.pytorch.pytorch_device_config import (
    get_working_device,
)
from model_compression_toolkit.core.pytorch.utils import set_model


class ConstantTensorIndexingModel(nn.Module):
    """Add a constant bias selected by an integer buffer index tuple."""

    indices: torch.Tensor

    def __init__(self, indices: list[int] | list[list[int]]) -> None:
        """Register the source Parameter and integer index buffer."""
        super().__init__()
        self.bias = nn.Parameter(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add the indexed bias to a runtime input."""
        return inputs + self.bias[:, self.indices]


class RuntimeTensorIndexingModel(nn.Module):
    """Index an upstream multiplication result using an integer buffer."""

    indices: torch.Tensor

    def __init__(self, indices: list[int] | list[list[int]]) -> None:
        """Register the integer indices."""
        super().__init__()
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Index the computed bias and add a floating-point constant."""
        bias = inputs * 2
        return bias[:, self.indices] + 0.25


def representative_data_gen(
    shape: tuple[int, ...],
) -> Iterator[list[torch.Tensor]]:
    """Yield reproducible floating-point calibration inputs."""
    size = prod(shape)
    source = torch.arange(
        size, dtype=torch.float32, device=get_working_device()
    ).reshape(shape) / size
    for offset in range(3):
        yield [source + offset / 4]


class TestQuantizationForGetitemTupleIndexing:
    """Verify the full PTQ path for direct integer index tuples."""

    @pytest.mark.parametrize(
        "indices, shape, expected_values, runtime_bias",
        [
            pytest.param(
                [2, 0], (2, 2),
                [
                    [[2, 0.25], [5.5, 3.75]],
                    [[2.25, 0.5], [5.75, 4]],
                ],
                False,
                id="input + bias[:, indices] (1D indices)",
            ),
            pytest.param(
                [[2, 0], [1, 2]], (2, 2, 2),
                [
                    [[[2, 0.125], [1.25, 2.375]],
                     [[5.5, 3.625], [4.75, 5.875]]],
                    [[[2.25, 0.375], [1.5, 2.625]],
                     [[5.75, 3.875], [5, 6.125]]],
                ],
                False,
                id="input + bias[:, indices] (2D indices)",
            ),
            pytest.param(
                [2, 0], (2, 4),
                [
                    [[0.75, 0.25], [1.75, 1.25]],
                    [[1.25, 0.75], [2.25, 1.75]],
                ],
                True,
                id="(input * 2)[:, indices] + 0.25 (1D indices)",
            ),
            pytest.param(
                [[2, 0], [1, 2]], (2, 4),
                [
                    [[[0.75, 0.25], [0.5, 0.75]],
                     [[1.75, 1.25], [1.5, 1.75]]],
                    [[[1.25, 0.75], [1, 1.25]],
                     [[2.25, 1.75], [2, 2.25]]],
                ],
                True,
                id="(input * 2)[:, indices] + 0.25 (2D indices)",
            ),
        ],
    )
    def test_ptq(
        self,
        indices: list[int] | list[list[int]],
        shape: tuple[int, ...],
        expected_values: list[list[list[float]] | list[list[list[float]]]],
        runtime_bias: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Check restored indices and output quantization through public PTQ."""
        if runtime_bias:
            model = RuntimeTensorIndexingModel(indices)
            references = {}
        else:
            model = ConstantTensorIndexingModel(indices)
            references = {0: model.bias.detach().cpu().numpy().copy()}
        set_model(model)
        indices_reference = model.indices.cpu().numpy().copy()
        references[1] = indices_reference
        output_shape = (2, *model.indices.shape)
        data_gen = partial(representative_data_gen, shape)
        inputs = [batch[0] for batch in data_gen()]
        assert len(inputs) == 3
        assert len(expected_values) == 2
        for original, repeated in zip(inputs, data_gen()):
            torch.testing.assert_close(
                original, repeated[0], rtol=0, atol=0
            )
        with torch.no_grad():
            for tensor, values in zip(inputs, expected_values):
                expected = torch.tensor(
                    values, dtype=torch.float32,
                    device=get_working_device(),
                )
                torch.testing.assert_close(
                    model(tensor), expected, rtol=0, atol=0
                )

        tpc = mct.get_target_platform_capabilities()
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=model,
            representative_data_gen=data_gen,
            target_platform_capabilities=tpc,
        )
        assert isinstance(quantized_model, pytorch_model_builder.PytorchModel)
        graph = quantized_model.graph
        getitem_nodes = [
            node for node in graph.nodes
            if isinstance(node, FunctionalNode)
            and node.functional_op is getitem
        ]
        add_nodes = [
            node for node in graph.nodes
            if isinstance(node, FunctionalNode) and node.functional_op is add
        ]
        assert len(getitem_nodes) == len(add_nodes) == 1
        getitem_node = getitem_nodes[0]
        add_node = add_nodes[0]
        assert any(
            edge.source_node is getitem_node
            for edge in graph.incoming_edges(add_node)
        )
        assert graph.get_outputs()[0].node is add_node
        assert set(getitem_node.weights) == set(references)
        incoming_edges = graph.incoming_edges(getitem_node)
        if runtime_bias:
            assert len(incoming_edges) == 1
            edge = incoming_edges[0]
            assert edge.sink_index == 0
            assert isinstance(edge.source_node, FunctionalNode)
            assert edge.source_node.functional_op is mul
        else:
            assert len(incoming_edges) == 0
        assert getitem_node.tensor_input_allocs == [0, (1, 1)]
        assert getitem_node.op_call_args == [(slice(None), None)]

        original_getitem = getattr(quantized_model, getitem_node.name)
        assert original_getitem is getitem
        getitem_outputs: list[torch.Tensor] = []

        def checked_getitem(
            bias: torch.Tensor,
            index: tuple[slice, torch.Tensor],
        ) -> torch.Tensor:
            """Observe restored arguments and run the real indexing operation."""
            assert isinstance(bias, torch.Tensor)
            assert bias.dtype == torch.float32
            assert tuple(bias.shape) == ((2, 4) if runtime_bias else (2, 3))
            assert isinstance(index, tuple) and len(index) == 2
            assert index[0] == slice(None)
            restored_indices = index[1]
            assert isinstance(restored_indices, torch.Tensor)
            assert restored_indices.dtype == torch.long
            assert tuple(restored_indices.shape) == indices_reference.shape
            np.testing.assert_array_equal(
                restored_indices.cpu().numpy(), indices_reference
            )
            expected = bias.index_select(
                1, restored_indices.flatten()
            ).reshape(output_shape)
            result = original_getitem(bias, index)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            getitem_outputs.append(result)
            return result

        monkeypatch.setattr(
            quantized_model, getitem_node.name, checked_getitem
        )
        assert add_node.is_activation_quantization_enabled()
        holder_name = quantized_model.node_to_activation_quantization_holder[
            add_node.name
        ]
        holder = quantized_model.get_submodule(holder_name)
        assert type(holder) is PytorchActivationQuantizationHolder
        assert holder.activation_holder_quantizer.num_bits == 8
        holder_outputs: list[torch.Tensor] = []

        def record_holder_output(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            """Record output quantization calls during inference."""
            assert module is holder
            assert len(args) == 1
            holder_outputs.append(output)

        handle = holder.register_forward_hook(record_holder_output)
        quantized_model.eval()
        try:
            with torch.no_grad():
                for position, tensor in enumerate(inputs[:2]):
                    output = quantized_model(tensor)
                    assert isinstance(output, torch.Tensor)
                    assert tuple(output.shape) == output_shape
                    assert output.dtype == torch.float32
                    assert torch.isfinite(output).all()
                    assert len(getitem_outputs) == position + 1
                    assert len(holder_outputs) == position + 1
                    assert holder_outputs[-1] is output
                    for key, reference in references.items():
                        saved = getitem_node.weights[key]
                        assert isinstance(saved, np.ndarray)
                        assert saved.shape == reference.shape
                        assert saved.dtype == reference.dtype
                        np.testing.assert_array_equal(saved, reference)
                    assert getitem_node.tensor_input_allocs == [0, (1, 1)]
                    assert getitem_node.op_call_args == [(slice(None), None)]
        finally:
            handle.remove()
        assert len(getitem_outputs) == 2
        assert len(holder_outputs) == 2
        assert model.indices.dtype == torch.long
        if isinstance(model, ConstantTensorIndexingModel):
            np.testing.assert_array_equal(
                model.bias.detach().cpu().numpy(), references[0]
            )
        np.testing.assert_array_equal(
            model.indices.cpu().numpy(), indices_reference
        )
