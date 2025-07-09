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
from unittest.mock import Mock
from typing import List
import torch
from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder, fully_quantized_wrapper
from mct_quantizers import QuantizationMethod, PytorchActivationQuantizationHolder, PytorchFLNActivationQuantizationHolder

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import CandidateNodeQuantizationConfig, NodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, Signedness
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.framework_info import ChannelAxisMapping
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import FrameworkQuantizationCapabilities
from model_compression_toolkit.core.pytorch.default_framework_info import PyTorchInfo
from model_compression_toolkit.core.common.framework_info import set_fw_info
from tests_pytest._test_util.graph_builder_utils import DummyLayer
from tests_pytest._test_util.tpc_util import minimal_tpc


def build_node(name='node', framework_attr={}, layer_class=DummyLayer,
               qcs: List[CandidateNodeQuantizationConfig] = None):
    node = BaseNode(name=name,
                    framework_attr=framework_attr,
                    input_shape=(4, 5, 6),
                    output_shape=(4, 5, 6),
                    weights={},
                    layer_class=layer_class,
                    reuse=False)
    if qcs:
        assert isinstance(qcs, list)
        node.quantization_cfg = NodeQuantizationConfig(base_quantization_cfg=qcs[0], candidates_quantization_cfg=qcs)
    return node

def build_qc(q_mode=ActivationQuantizationMode.QUANT):
    op_cfg = OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        supported_input_activation_n_bits=8,
        signedness=Signedness.AUTO
    )
    a_qcfg = NodeActivationQuantizationConfig(op_cfg=op_cfg)
    a_qcfg.quant_mode = q_mode
    w_qcfg = NodeWeightsQuantizationConfig(op_cfg=op_cfg,
                                           weights_channels_axis=ChannelAxisMapping(0, 1),
                                           node_attrs_list=['weight', 'bias'])
    qc = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qcfg, 
                                         weights_quantization_cfg=w_qcfg)
    return qc

def get_test_graph():
    
    set_fw_info(PyTorchInfo)

    conv1 = build_node('conv1', framework_attr={'in_channels':3, 'out_channels':3, 'kernel_size':3}, 
                       layer_class=torch.nn.Conv2d, qcs=[build_qc(q_mode=ActivationQuantizationMode.FLN_QUANT)])
    relu = build_node('relu', layer_class=torch.nn.ReLU, qcs=[build_qc()])
    conv2 = build_node('conv2', framework_attr={'in_channels':3, 'out_channels':3, 'kernel_size':3}, 
                       layer_class=torch.nn.Conv2d, qcs=[build_qc(q_mode=ActivationQuantizationMode.FLN_NO_QUANT)])
    sigmoid = build_node('sigmoid', layer_class=torch.nn.Sigmoid, qcs=[build_qc()])
    flatten = build_node('flatten', layer_class=torch.nn.Flatten, 
                         qcs=[build_qc(q_mode=ActivationQuantizationMode.PRESERVE_QUANT)])
    fc = build_node('fc', framework_attr={'in_features':48, 'out_features':10}, 
                    layer_class=torch.nn.Linear, qcs=[build_qc(q_mode=ActivationQuantizationMode.FLN_QUANT)])
    hswish = build_node('hswish', layer_class=torch.nn.Hardswish, qcs=[build_qc()])
    
    graph = Graph('g', input_nodes=[conv1],
                  nodes=[relu, conv2, sigmoid, flatten, fc],
                  output_nodes=[hswish],
                  edge_list=[Edge(conv1, relu, 0, 0),
                             Edge(relu, conv2, 0, 0),
                             Edge(conv2, sigmoid, 0, 0),
                             Edge(sigmoid, flatten, 0, 0),
                             Edge(flatten, fc, 0, 0),
                             Edge(fc, hswish, 0, 0),
                            ]
                )
    fqc = FrameworkQuantizationCapabilities(tpc=minimal_tpc(), name="test")
    graph.set_fqc(fqc)

    return graph

def get_inferable_quantizers_mock(node):

    if node.name == 'conv2' or node.name == 'relu':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 8
        activation_quantizers.signed = False
        activation_quantizers.threshold_np = 8.0
    
    elif node.name == 'conv1' or node.name == 'fc':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 16
        activation_quantizers.signed = True
        activation_quantizers.threshold_np = 16.0
        
    elif node.name == 'sigmoid' or node.name == 'hswish':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 4
        activation_quantizers.signed = False
        activation_quantizers.threshold_np = 4.0
    else:
        return {}, []
    
    return {}, [activation_quantizers]


class TestPyTorchModelBuilder():

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
        
        # check conv1
        assert hasattr(exportable_model, 'conv1_activation_holder_quantizer')
        conv1_activation_holder_quantizer = exportable_model.conv1_activation_holder_quantizer
        assert isinstance(conv1_activation_holder_quantizer, PytorchFLNActivationQuantizationHolder)
        assert conv1_activation_holder_quantizer.quantization_bypass == True
        assert conv1_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert conv1_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert conv1_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0

        # check relu
        assert hasattr(exportable_model, 'relu_activation_holder_quantizer')
        relu_activation_holder_quantizer = exportable_model.relu_activation_holder_quantizer
        assert isinstance(relu_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert relu_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8
        assert relu_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert relu_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 8.0

        # check conv2 (FLN_NO_QUANT)
        assert not hasattr(exportable_model, 'conv2_activation_holder_quantizer')
 
        # check sigmoid
        assert hasattr(exportable_model, 'sigmoid_activation_holder_quantizer')
        sigmoid_activation_holder_quantizer = exportable_model.sigmoid_activation_holder_quantizer
        assert isinstance(sigmoid_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert sigmoid_activation_holder_quantizer.activation_holder_quantizer.num_bits == 4
        assert sigmoid_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert sigmoid_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 4.0

        # check fc
        assert hasattr(exportable_model, 'fc_activation_holder_quantizer')
        fc_activation_holder_quantizer = exportable_model.fc_activation_holder_quantizer
        assert isinstance(fc_activation_holder_quantizer, PytorchFLNActivationQuantizationHolder)
        assert fc_activation_holder_quantizer.quantization_bypass == True
        assert fc_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert fc_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert fc_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0

        # check hswish
        assert hasattr(exportable_model, 'hswish_activation_holder_quantizer')
        hswish_activation_holder_quantizer = exportable_model.hswish_activation_holder_quantizer
        assert isinstance(hswish_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert hswish_activation_holder_quantizer.activation_holder_quantizer.num_bits == 4
        assert hswish_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert hswish_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 4.0
