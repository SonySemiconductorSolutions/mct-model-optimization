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
import pytest
from tests_pytest._test_util.graph_builder_utils import build_node
from unittest.mock import Mock
import numpy as np
import torch
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy
from model_compression_toolkit.core.common import StatsCollector, Graph
from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.defaultdict import DefaultDict


class Conv2D:
    pass

class Linear:
    pass

class ConvTranspose2d:
    pass

class DummyLayer:
    pass

@pytest.fixture
def fw_impl_mock():
    fw_impl = Mock()
    fw_impl.model_builder.return_value = (Mock(), None)
    return fw_impl

@pytest.fixture
def fw_info_mock():
    fw_info = Mock()
    fw_info.out_channel_axis_mapping = DefaultDict({Conv2D: 1, Linear: -1, ConvTranspose2d: 1}, 1)
    return fw_info


class TestModelCollectorInit:

    def test_init(self, fw_impl_mock, fw_info_mock):
        node0 = build_node('node0', output_shape=[[1, 3, 2, 2]])   # 4D tensor
        node1 = build_node('node1', output_shape=[[3, 2]]) # 2D tensor
        node2 = build_node('node2', output_shape=[[4]])   # 1D tensor
        node3 = build_node('node3', output_shape=[[]])     # Scalar

        mock_nodes_list = [node0, node1, node2, node3]
        for node in mock_nodes_list:
            node.is_activation_quantization_enabled = Mock(return_value=True)
            node.is_fln_quantization = Mock(return_value=False)

        graph = Graph('g',
                      input_nodes=[node0],
                      nodes=mock_nodes_list,
                      output_nodes=[OutTensor(node3, 0)],
                      edge_list=[Edge(node0, node1, 0, 0), Edge(node1, node2, 0, 0), Edge(node2, node3, 0, 0)])
        graph.set_out_stats_collector_to_node = Mock(wraps=graph.set_out_stats_collector_to_node)

        fw_info_mock.get_kernel_op_attributes.return_value = [None]

        mc = ModelCollector(graph, fw_impl_mock, fw_info_mock)

        # If output shape is scalar or 1D tensor, the axis should be -1.
        # If output shape is 2D tensor, the axis should be 1.
        expected_axis = [1, 1, -1, -1]
        for node, expected in zip(graph.nodes, expected_axis):
            out_stats_container = graph.get_out_stats_collector(node)
            assert isinstance(out_stats_container, StatsCollector)

            assert out_stats_container.mpcc.axis == expected
            assert out_stats_container.mc.axis == expected


class TestModelCollectorInfer:

    def test_infer(self, fw_impl_mock, fw_info_mock):
        node0 = build_node('node0', output_shape=[[1, 3, 2, 2]])   # 4D tensor
        node1 = build_node('node1', output_shape=[[3, 2]])   # 2D tensor
        node2 = build_node('node2', output_shape=[[4]])     # 1D tensor
        node3 = build_node('node3', output_shape=[[]])       # scalar

        mock_nodes_list = [node0, node1, node2, node3]
        for node in mock_nodes_list:
            node.is_activation_quantization_enabled = Mock(return_value=True)
            node.is_fln_quantization = Mock(return_value=False)

        graph = Graph('g',
                      input_nodes=[node0],
                      nodes=mock_nodes_list,
                      output_nodes=[OutTensor(node3, 0)],
                      edge_list=[Edge(node0, node1, 0, 0), Edge(node1, node2, 0, 0), Edge(node2, node3, 0, 0)])

        fw_info_mock.get_kernel_op_attributes.return_value = [None]

        infer1 = [
            torch.tensor(
                [[
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[-1.0, -2.0], [-3.0, -4.0]],
                    [[10.0, 10.0], [10.0, 10.0]],
                ]],
                dtype=torch.float32,
            ),
            torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]], dtype=torch.float32),
            torch.tensor([2.0, 6.0, 8.0, 10.0], dtype=torch.float32),
            torch.tensor(10.0, dtype=torch.float32),
        ]
        infer2 = [
            torch.tensor(
                [[
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[-10.0, -20.0], [-30.0, -40.0]],
                ]],
                dtype=torch.float32,
            ),
            torch.tensor([[5.0, -1.0], [6.0, -2.0], [7.0, -3.0]], dtype=torch.float32),
            torch.tensor([4.0, 8.0, 12.0, 16.0], dtype=torch.float32),
            torch.tensor(-2.0, dtype=torch.float32),
        ]

        fw_impl_mock.to_numpy.side_effect = torch_tensor_to_numpy
        fw_impl_mock.run_model_inference.side_effect = [infer1, infer2]

        mc = ModelCollector(graph, fw_impl_mock, fw_info_mock)

        dummy_input = [np.random.randn(1, 3, 2, 2)]
        mc.infer(dummy_input)
        mc.infer(dummy_input)

        sc0 = graph.get_out_stats_collector(node0)
        sc1 = graph.get_out_stats_collector(node1)
        sc2 = graph.get_out_stats_collector(node2)
        sc3 = graph.get_out_stats_collector(node3)

        # node0 (axis=1)
        # infer1 channel means: [2.5, -2.5, 10.0]
        # infer2 channel means: [6.5,  1.5, -25.0]
        # final mean: [4.5, -0.5, -7.5]
        np.testing.assert_allclose(sc0.get_mean(), np.array([4.5, -0.5, -7.5]))
        min_v, max_v = sc0.get_min_max_values()
        assert min_v == -40.0
        assert max_v == 10.0
        
        # node1 (axis=1)
        # infer1 channel means: [2, 4]
        # infer2 channel means: [6, -2]
        # final mean: [4, 1]
        np.testing.assert_allclose(sc1.get_mean(), np.array([4.0, 1.0]))
        min_v, max_v = sc1.get_min_max_values()
        assert min_v == -3.0
        assert max_v == 7.0
        
        # node2 (axis=-1)
        # infer1 channel means: [2, 6, 8, 10]
        # infer2 channel means: [4, 8, 12, 16]
        # final mean: [3, 7, 10, 13]
        np.testing.assert_allclose(sc2.get_mean(), np.array([3.0, 7.0, 10.0, 13.0]))
        min_v, max_v = sc2.get_min_max_values()
        assert min_v == 2.0
        assert max_v == 16.0
        
        # node3 (axis=-1)
        # infer1 channel means: 10
        # infer2 channel means: -2
        # final mean: 4
        np.testing.assert_allclose(sc3.get_mean(), np.array([4.0]))
        min_v, max_v = sc3.get_min_max_values()
        assert min_v == -2.0
        assert max_v == 10.0