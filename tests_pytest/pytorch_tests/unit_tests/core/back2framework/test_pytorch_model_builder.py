# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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

import pytest
from unittest.mock import Mock
from typing import List
import torch
from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder, fully_quantized_wrapper
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from tests_pytest._test_util.graph_builder_utils import DummyLayer
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness
from model_compression_toolkit.core import QuantizationConfig
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder, _merge_inputs
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema

def build_node(name='node', framework_attr={}, qcs: List[CandidateNodeQuantizationConfig] = None,
               input_shape=(4, 5, 6), output_shape=(4, 5, 6), weights = {},
               layer_class=DummyLayer, reuse=False):

    node = BaseNode(name=name,
                    framework_attr=framework_attr,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    weights=weights,
                    layer_class=layer_class,
                    reuse=reuse)
    if qcs:
        assert isinstance(qcs, list)
        node.candidates_quantization_cfg = qcs
        node.final_activation_quantization_cfg = node.candidates_quantization_cfg[0].activation_quantization_cfg
    return node

def build_qc(a_nbits=8, a_enable=True, q_preserving=False, aq_params={}):
    op_cfg = OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=a_nbits,
        enable_activation_quantization=a_enable,
        quantization_preserving=q_preserving,
        supported_input_activation_n_bits=8,
        signedness=Signedness.AUTO
    )
    a_qcfg = NodeActivationQuantizationConfig(qc=QuantizationConfig(), op_cfg=op_cfg,
                                              activation_quantization_fn=None,
                                              activation_quantization_params_fn=None)
    if len(aq_params) != 0:
        a_qcfg.set_activation_quantization_param(aq_params)
    qc = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qcfg)
    return qc

def get_tpc():
    base_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)
    
    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    mixed_precision_cfg_list = [base_config]

    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list), 
                                                                             base_config=base_config)
    operator_set = []
    preserving_quantization_config = (default_configuration_options.clone_and_edit(enable_activation_quantization=False, quantization_preserving=True))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.DROPOUT, qc_options=preserving_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.FLATTEN, qc_options=preserving_quantization_config))
    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    fc = schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED, qc_options=mixed_precision_configuration_options)

    operator_set.extend([conv, fc])
    tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set),
        insert_preserving_quantizers=True)
    return tpc

# test graph
def get_test_graph():

    conv = build_node('conv', framework_attr={'in_channels':3, 'out_channels':3, 'kernel_size':3}, layer_class=torch.nn.Conv2d, qcs=[build_qc()])
    dropout = build_node('dropout', layer_class=torch.nn.Dropout, qcs=[build_qc(a_enable=False, q_preserving=True)])
    flatten1 = build_node('flatten1', layer_class=torch.nn.Flatten, qcs=[build_qc(a_enable=False, q_preserving=True)])
    flatten2 = build_node('flatten2', layer_class=torch.nn.Flatten, qcs=[build_qc(a_enable=False, q_preserving=True)])
    fc1 = build_node('fc1', framework_attr={'in_features':48, 'out_features':128}, layer_class=torch.nn.Linear, qcs=[build_qc()])
    fc2 = build_node('fc2', framework_attr={'in_features':48, 'out_features':128}, layer_class=torch.nn.Linear, qcs=[build_qc()])

    graph = Graph('g', input_nodes=[conv],
                  nodes=[flatten1, fc1, flatten2, dropout],
                  output_nodes=[fc2],
                  edge_list=[Edge(conv, flatten1, 0, 0),
                             Edge(flatten1, fc1, 0, 0),
                             Edge(fc1, flatten2, 0, 0),
                             Edge(flatten2, dropout, 0, 0),
                             Edge(dropout, fc2, 0, 0),
                             ]
                  )
    tpc = get_tpc()
    fqc = FrameworkQuantizationCapabilities(tpc=tpc, name="test")
    graph.set_fqc(fqc)

    return graph

def get_inferable_quantizers_mock(node):
    if node.name == 'conv':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 8
        activation_quantizers.signed = False
        activation_quantizers.threshold_np = 8.0
    
    elif node.name == 'fc1':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 16
        activation_quantizers.signed = True
        activation_quantizers.threshold_np = 16.0
        
    elif node.name == 'fc2':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 4
        activation_quantizers.signed = False
        activation_quantizers.threshold_np = 4.0
    else:
        return {}, []
    
    return {}, [activation_quantizers]

def build_getitem_functional_node(
    allocations: list[int | tuple[int, int]],
    call_args: list[Any],
) -> FunctionalNode:
    """Build an operator.getitem node for index-tuple reconstruction tests."""
    return FunctionalNode(
        name="operation",
        framework_attr={},
        input_shape=(),
        output_shape=(),
        weights={},
        layer_class=type(getitem),
        op_call_args=tuple(call_args),
        op_call_kwargs={},
        functional_op=getitem,
        tensor_input_allocs=allocations,
    )


class TestPyTorchModelBuilder():
    @pytest.mark.parametrize(
        "shape, saved_index, allocations, index_values, expected_values",
        [
            pytest.param((2, 3), (slice(None), None), [0, (1, 1)],
                         [[2, 0]], [[2, 0], [5, 3]], id="column"),
            pytest.param((2, 3), (None, slice(None)), [0, (1, 0)],
                         [[1, 0]], [[3, 4, 5], [0, 1, 2]], id="row"),
            pytest.param(
                (2, 3, 4), (slice(None), None, slice(None)), [0, (1, 1)],
                [[2, 0]],
                [[[8, 9, 10, 11], [0, 1, 2, 3]],
                 [[20, 21, 22, 23], [12, 13, 14, 15]]], id="middle",
            ),
            pytest.param(
                (2, 3, 4), (Ellipsis, None), [0, (1, 1)], [[2, 0]],
                [[[2, 0], [6, 4], [10, 8]],
                 [[14, 12], [18, 16], [22, 20]]], id="ellipsis",
            ),
            pytest.param((2, 3), (None, None), [0, (1, 0), (1, 1)],
                         [[1, 0], [2, 0]], [5, 0], id="pair"),
            pytest.param((2, 3), (None,), [0, (1, 0)],
                         [[1, 0]], [[3, 4, 5], [0, 1, 2]], id="singleton"),
            pytest.param((2, 3), (slice(None), 0), [0], [],
                         [0, 3], id="fixed_tuple"),
            pytest.param((2, 3), (None, slice(None), None), [0, (1, 2)],
                         [[2, 0]], [[[2, 0], [5, 3]]], id="newaxis"),
        ],
    )
    def test_merge_getitem_indices(
        self,
        shape: tuple[int, ...],
        saved_index: tuple[Any, ...],
        allocations: list[int | tuple[int, int]],
        index_values: list[list[int]],
        expected_values: list[Any],
    ) -> None:
        """Restore operator.getitem index tuples and verify exact outputs."""
        x = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
        indices = [torch.tensor(values, dtype=torch.long)
                   for values in index_values]
        node = build_getitem_functional_node(allocations, [saved_index])
        saved_args = node.op_call_args.copy()
        saved_allocations = node.tensor_input_allocs.copy()

        merged, kwargs = _merge_inputs(
            node, [x, *indices], node.op_call_args, {}
        )

        assert merged[0] is x
        assert kwargs == {}
        for allocation, tensor in zip(allocations[1:], indices):
            if isinstance(allocation, tuple):
                assert isinstance(merged[allocation[0]], tuple)
                assert merged[allocation[0]][allocation[1]] is tensor
            else:
                assert merged[allocation] is tensor
        for position, fixed_value in enumerate(saved_index):
            if (1, position) not in allocations:
                assert merged[1][position] == fixed_value
        expected = torch.tensor(expected_values, dtype=torch.float32)
        result = getitem(*merged)
        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert node.op_call_args == saved_args
        assert node.tensor_input_allocs == saved_allocations

    def test_getitem_deferred_tuple_restore(self) -> None:
        """Restore operator.getitem index tuples after top-level insertion."""
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        cols = torch.tensor([2, 0], dtype=torch.long)
        node = build_getitem_functional_node([0, (1, 1)], [(slice(None), None)])

        merged, kwargs = _merge_inputs(
            node, [cols, x], node.op_call_args, {},
            tensor_input_allocs=[(1, 1), 0],
        )

        assert merged[0] is x
        assert merged[1][0] == slice(None)
        assert merged[1][1] is cols
        assert kwargs == {}
        torch.testing.assert_close(
            getitem(*merged), torch.tensor([[2., 0.], [5., 3.]]),
            rtol=0, atol=0,
        )

    def test_getitem_saved_args_reused(self) -> None:
        """Reuse saved operator.getitem index tuples with fresh tensors."""
        node = build_getitem_functional_node([0, (1, 1)], [(slice(None), None)])
        original_index = node.op_call_args[0]
        first_x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        first_cols = torch.tensor([2, 0], dtype=torch.long)
        second_x = first_x + 10
        second_cols = torch.tensor([1, 2], dtype=torch.long)

        first, _ = _merge_inputs(
            node, [first_x, first_cols], node.op_call_args, {}
        )
        second, _ = _merge_inputs(
            node, [second_x, second_cols], node.op_call_args, {}
        )

        assert first[0] is first_x
        assert first[1][1] is first_cols
        assert second[0] is second_x
        assert second[1][1] is second_cols
        assert node.op_call_args == [(slice(None), None)]
        assert node.op_call_args[0] is original_index
        assert node.tensor_input_allocs == [0, (1, 1)]
        torch.testing.assert_close(
            getitem(*first), torch.tensor([[2., 0.], [5., 3.]]),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            getitem(*second), torch.tensor([[11., 12.], [14., 15.]]),
            rtol=0, atol=0,
        )

    # test case for PyTorchModelBuilder
    def test_pytorch_model(self, fw_impl_mock):
        graph = get_test_graph()
        fw_impl_mock.get_inferable_quantizers.side_effect = lambda node: get_inferable_quantizers_mock(node)
        exportable_model, _ = PyTorchModelBuilder(graph=graph,
                                                  wrapper=lambda n, m:
                                                  fully_quantized_wrapper(n, m,
                                                                          fw_impl=fw_impl_mock),
                                                  get_activation_quantizer_holder_fn=lambda n, holder_type, **kwargs:
                                                  get_activation_quantizer_holder(n, holder_type,
                                                                                  fw_impl=fw_impl_mock, **kwargs)).build_model()
        
        # check conv
        conv_activation_holder_quantizer = exportable_model.conv_activation_holder_quantizer
        assert isinstance(conv_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert conv_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8
        assert conv_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert conv_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 8.0

        # check flatten1 (same conv)
        flatten1_activation_holder_quantizer = exportable_model.flatten1_activation_holder_quantizer
        assert isinstance(flatten1_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
        assert flatten1_activation_holder_quantizer.quantization_bypass == True
        assert flatten1_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8
        assert flatten1_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert flatten1_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 8.0

        # check fc1
        fc1_activation_holder_quantizer = exportable_model.fc1_activation_holder_quantizer
        assert isinstance(fc1_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert fc1_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert fc1_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert fc1_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0

        # check flatten2 (same fc1)
        flatten2_activation_holder_quantizer = exportable_model.flatten2_activation_holder_quantizer
        assert isinstance(flatten2_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
        assert flatten2_activation_holder_quantizer.quantization_bypass == True
        assert flatten2_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert flatten2_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert flatten2_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0
        
        # check dropout (same fc1)
        dropout_activation_holder_quantizer = exportable_model.dropout_activation_holder_quantizer
        assert isinstance(dropout_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
        assert dropout_activation_holder_quantizer.quantization_bypass == True
        assert dropout_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert dropout_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert dropout_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0

        # check fc2
        fc2_activation_holder_quantizer = exportable_model.fc2_activation_holder_quantizer
        assert isinstance(fc2_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert fc2_activation_holder_quantizer.activation_holder_quantizer.num_bits == 4
        assert fc2_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert fc2_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 4.0
