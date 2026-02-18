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
from unittest.mock import Mock
import pytest
from model_compression_toolkit.core.common.model_collector import create_stats_collector_for_node
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
def fw_info_mock():
    fw_info = Mock()
    fw_info.out_channel_axis_mapping = DefaultDict({Conv2D: 1, Linear: -1, ConvTranspose2d: 1}, 1)
    return fw_info

@pytest.fixture
def node_mock():
    node = Mock()
    node.is_activation_quantization_enabled.return_value = True
    return node


class TestCreateStatsCollectorForNode:

    def test_create_stats_collector_for_node_conv(self, node_mock, fw_info_mock):
        node_mock.type = Conv2D
        node_mock.get_output_shapes_list.return_value = [[1, 3, 32, 32]]

        assert fw_info_mock.out_channel_axis_mapping.get(node_mock.type) == 1
        collector = create_stats_collector_for_node(node_mock, fw_info_mock, quant_node_in_fln=False)
        assert collector.mc.axis == 1
        assert collector.mpcc.axis == 1

    def test_create_stats_collector_for_node_linear(self, node_mock, fw_info_mock):
        node_mock.type = Linear
        node_mock.get_output_shapes_list.return_value = [[1, 10]]

        assert fw_info_mock.out_channel_axis_mapping.get(node_mock.type) == -1
        collector = create_stats_collector_for_node(node_mock, fw_info_mock, quant_node_in_fln=False)
        assert collector.mc.axis == -1
        assert collector.mpcc.axis == -1

    def test_create_stats_collector_for_node_2d_tensor(self, node_mock, fw_info_mock):
        node_mock.type = DummyLayer
        node_mock.get_output_shapes_list.return_value = [[1, 3]] # Output shape is 2D tensor

        assert fw_info_mock.out_channel_axis_mapping.get(node_mock.type) == 1
        collector = create_stats_collector_for_node(node_mock, fw_info_mock, quant_node_in_fln=False)
        assert collector.mc.axis == 1
        assert collector.mpcc.axis == 1

    def test_create_stats_collector_for_node_1d_tensor(self, node_mock, fw_info_mock):
        node_mock.type = DummyLayer
        node_mock.get_output_shapes_list.return_value = [[1]] # Output shape is 1D tensor

        # Check that out_channel_axis_mapping changed to -1
        assert fw_info_mock.out_channel_axis_mapping.get(node_mock.type) == 1
        collector = create_stats_collector_for_node(node_mock, fw_info_mock, quant_node_in_fln=False)
        assert collector.mc.axis == -1
        assert collector.mpcc.axis == -1

    def test_create_stats_collector_for_node_scalar(self, node_mock, fw_info_mock):
        node_mock.type = DummyLayer
        node_mock.get_output_shapes_list.return_value = [[]] # Output shape is scalar

        # Check that out_channel_axis_mapping changed to -1
        assert fw_info_mock.out_channel_axis_mapping.get(node_mock.type) == 1
        collector = create_stats_collector_for_node(node_mock, fw_info_mock, quant_node_in_fln=False)
        assert collector.mc.axis == -1
        assert collector.mpcc.axis == -1