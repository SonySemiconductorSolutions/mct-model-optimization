#  Copyright 2025 Sony Semiconductor Solutions. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
import pytest
from unittest.mock import Mock
from typing import List, Tuple
import os
import tempfile
from functools import partial
import shutil
import numpy as np

import torch

import model_compression_toolkit as mct
from model_compression_toolkit.xquant.pytorch.judge_troubleshoot_utils import judge_mixed_precision_with_model_output_loss_objective, judge_outlier_removal, judge_shift_negative_activation, judge_unbalanced_concatnation
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig
from mct_quantizers.pytorch.activation_quantization_holder import PytorchActivationQuantizationHolder
from mct_quantizers.pytorch.quantize_wrapper import PytorchQuantizationWrapper
from model_compression_toolkit.core.common.collectors.statistics_collector import StatsCollector
from model_compression_toolkit.logger import Logger

from tests.pytorch_tests.xquant_tests.test_xquant_end2end import random_data_gen
from tests_pytest._test_util.graph_builder_utils import build_node


# check for judge_outlier_removal
@pytest.mark.parametrize(("inputs", "expected"), [
    (100, []),
    (0.01, [('conv1', 'dummy_report_dir/outlier_histgrams/conv1.png')]),
])
def test_judge_outlier_removal(inputs, expected):

    conv1 = build_node('conv1', layer_class=torch.nn.Conv2d)
    conv1_stats_collector = StatsCollector(out_channel_axis=1)
    conv1_stats_collector.hc._n_bins = 11
    conv1_stats_collector.hc._bins = np.array([-4,-3,-2,-1,0,1,2,3,4,50,100])
    conv1_stats_collector.hc._counts = np.array([10,20,30,40,50,40,30,20,10,1])
    
    graph = Mock()
    graph.find_node_by_name.return_value = [conv1]
    graph.get_out_stats_collector.return_value = conv1_stats_collector
    
    xquant_config = XQuantConfig(report_dir='dummy_report_dir',
                                 threshold_zscore_outlier_removal=inputs)

    if(not os.path.exists('dummy_report_dir/outlier_histgrams')):
        os.makedirs('dummy_report_dir/outlier_histgrams')

    result = judge_outlier_removal(degrade_layers=['conv1'], float_graph=graph, xquant_config=xquant_config)
    
    assert isinstance(result, List)

    if len(result) == 0:
        assert result == expected
    else:
        assert isinstance(result[0], Tuple)
        assert result[0] == expected[0]

        shutil.rmtree('dummy_report_dir') # remove dummy_report_dir

# check for judge_outlier_removal(when output directory not found)
@pytest.mark.parametrize(("inputs", "expected"), [
    (0.01, [('conv1', 'dummy_report_dir/outlier_histgrams/conv1.png')]),
])
def test_judge_outlier_removal_outdir_not_found(inputs, expected):

    conv1 = build_node('conv1', layer_class=torch.nn.Conv2d)
    conv1_stats_collector = StatsCollector(out_channel_axis=1)
    conv1_stats_collector.hc._n_bins = 11
    conv1_stats_collector.hc._bins = np.array([-4,-3,-2,-1,0,1,2,3,4,50,100])
    conv1_stats_collector.hc._counts = np.array([10,20,30,40,50,40,30,20,10,1])
    
    graph = Mock()
    graph.find_node_by_name.return_value = [conv1]
    graph.get_out_stats_collector.return_value = conv1_stats_collector
    
    xquant_config = XQuantConfig(report_dir='dummy_report_dir',
                                 threshold_zscore_outlier_removal=inputs)

    if(os.path.exists('dummy_report_dir/outlier_histgrams')):
        shutil.rmtree('dummy_report_dir/outlier_histgrams')

    result = judge_outlier_removal(degrade_layers=['conv1'], float_graph=graph, xquant_config=xquant_config)
    
    assert isinstance(result, List)

    if len(result) == 0:
        assert result == expected
    else:
        assert isinstance(result[0], Tuple)
        assert result[0] == expected[0]


# check for judge_mixed_precision_with_model_output_loss_objective
@pytest.mark.parametrize(("inputs", "expected"), [
    # inputs: (weight_n_bits, activation_n_bits, threshold)
    ((8, 8, 2), []), 
    ((1, 1, 2), ['wq']),
])
def test_judge_mixed_precision_with_model_output_loss_objective(inputs, expected):

    model = Mock()

    weight_quantizer = Mock()
    weight_quantizer.num_bits = inputs[0]
    weight_quantization_holder = Mock(spec=PytorchQuantizationWrapper)
    weight_quantization_holder.weights_quantizers = {'weight': weight_quantizer}

    activation_holder_quantizer = Mock()
    activation_holder_quantizer.num_bits = inputs[1]
    activation_quantization_holder = Mock(spec=PytorchActivationQuantizationHolder)
    activation_quantization_holder.activation_holder_quantizer = activation_holder_quantizer

    model.named_children.return_value = [('wq', weight_quantization_holder), ('aq', activation_quantization_holder)]

    xquant_config = XQuantConfig(report_dir=None,
                                 threshold_bitwidth_mixed_precision_with_model_output_loss_objective=inputs[2])
    
    result = judge_mixed_precision_with_model_output_loss_objective(model, xquant_config)
    assert isinstance(result, List)
    assert result == expected

# check for judge_mixed_precision_with_model_output_loss_objective
@pytest.mark.parametrize(("inputs", "expected"), [
    # inputs: (weight_n_bits, activation_n_bits, threshold)
    ((1, 1, 2), ['wq']),
])
def test_judge_mixed_precision_with_model_output_loss_objective_no_weight_quantizer(inputs, expected):

    model = Mock()

    activation_holder_quantizer = Mock()
    activation_holder_quantizer.num_bits = inputs[1]
    activation_quantization_holder = Mock(spec=PytorchActivationQuantizationHolder)
    activation_quantization_holder.activation_holder_quantizer = activation_holder_quantizer

    model.named_children.return_value = [('wq', None), ('aq', activation_quantization_holder)]

    xquant_config = XQuantConfig(report_dir=None,
                                 threshold_bitwidth_mixed_precision_with_model_output_loss_objective=inputs[2])
    
    result = judge_mixed_precision_with_model_output_loss_objective(model, xquant_config)
    assert isinstance(result, List)
    assert result == expected

# check for judge_mixed_precision_with_model_output_loss_objective
@pytest.mark.parametrize(("inputs", "expected"), [
    # inputs: (weight_n_bits, activation_n_bits, threshold)
    ((1, 1, 2), ['wq']),
])
def test_judge_mixed_precision_with_model_output_loss_objective_no_activation_holder_quantizer(inputs, expected):

    model = Mock()

    weight_quantizer = Mock()
    weight_quantizer.num_bits = inputs[0]
    weight_quantization_holder = Mock(spec=PytorchQuantizationWrapper)
    weight_quantization_holder.weights_quantizers = {'weight': weight_quantizer}

    model.named_children.return_value = [('wq', weight_quantization_holder), ('aq', None)]

    xquant_config = XQuantConfig(report_dir=None,
                                 threshold_bitwidth_mixed_precision_with_model_output_loss_objective=inputs[2])
    
    result = judge_mixed_precision_with_model_output_loss_objective(model, xquant_config)
    assert isinstance(result, List)
    assert result == expected


# check for judge_shift_negative_activation
@pytest.mark.parametrize(("inputs", "expected"), [
    (('prelu', torch.nn.PReLU), ['prelu']),
    (('elu', torch.nn.ELU), ['elu']),
    (('hardswish', torch.nn.Hardswish), ['hardswish']),
    (('silu', torch.nn.SiLU), ['silu']),
    (('gelu', torch.nn.GELU), ['gelu']),
    (('prelu', torch.nn.PReLU, 'silu', torch.nn.SiLU), ['prelu', 'silu']),
    (('relu', torch.nn.ReLU), []),
])
def test_judge_shift_negative_activation(inputs, expected):

    conv1 = build_node('conv1', layer_class=torch.nn.Conv2d)

    if len(inputs) > 2:
        act1 = build_node(inputs[0], layer_class=inputs[1])
        conv2 = build_node('conv2', layer_class=torch.nn.Conv2d)
        act2 = build_node(inputs[2], layer_class=inputs[3])
        graph = Graph('g', input_nodes=[conv1],
                      nodes=[act1, conv2],
                      output_nodes=[act2],
                      edge_list=[Edge(conv1, act1, 0, 0),
                                 Edge(act1, conv2, 0, 0),
                                 Edge(conv2, act2, 0, 0)])
    else:
        act = build_node(inputs[0], layer_class=inputs[1])
        graph = Graph('g', input_nodes=[conv1],
                      nodes=[],
                      output_nodes=[act],
                      edge_list=[Edge(conv1, act, 0, 0)])
        
    result = judge_shift_negative_activation(graph, xquant_config=Mock())

    assert isinstance(result, List) 
    assert result == expected


# check for unbalanced concatnation(conv1x1 concat)
@pytest.mark.parametrize(("inputs", "expected"), [
    (1.0, [("conv1", "conv2")])
])
def test_judge_unbalanced_concatnation_conv1x1(inputs, expected):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    float_model = Model()

    def get_input_shape():
        return (3, 224, 224)
    
    tmpdir1 = tempfile.mkdtemp()
    repr_dataset = partial(random_data_gen, shape=get_input_shape())
    validation_dataset = partial(random_data_gen, use_labels=True)
    mct.set_log_folder(tmpdir1)
    mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                             representative_data_gen=repr_dataset)
   
    xquant_config = XQuantConfig(report_dir =  tmpdir1,
                                quantize_reported_dir = None,
                                threshold_ratio_unbalanced_concatenation = inputs
                                )
    # Initialize the logger with the report directory.
    result = judge_unbalanced_concatnation(degrade_layers=["conv1_conv2_collapsed"],
                                           float_model=float_model,
                                           dataset=repr_dataset,
                                           xquant_config=xquant_config
                                           )
    
    if(len(expected) > 0):
        for n in range(len(expected)):
            assert result[n][0] == expected[n][0]
            assert result[n][1] == expected[n][1]
    else:
        assert result == expected

# check for unbalanced concatnation(add concat)
@pytest.mark.parametrize(("inputs", "expected"), [
    (1.0, [("conv1", "add")])
])
def test_judge_unbalanced_concatnation_add(inputs, expected):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        def forward(self, x):
            x1 = self.conv1(x)
            x = torch.add(x, x1)
            return x

    float_model = Model()

    def get_input_shape():
        return (3, 224, 224)
    
    tmpdir1 = tempfile.mkdtemp()
    repr_dataset = partial(random_data_gen, shape=get_input_shape())
    validation_dataset = partial(random_data_gen, use_labels=True)
    mct.set_log_folder(tmpdir1)
    mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                             representative_data_gen=repr_dataset)
   
    xquant_config = XQuantConfig(report_dir =  tmpdir1,
                                quantize_reported_dir = None,
                                threshold_ratio_unbalanced_concatenation = inputs
                                )
    # Initialize the logger with the report directory.
    result = judge_unbalanced_concatnation(degrade_layers=["conv1_add_collapsed"],
                                           float_model=float_model,
                                           dataset=repr_dataset,
                                           xquant_config=xquant_config
                                           )
    
    if(len(expected) > 0):
        for n in range(len(expected)):
            assert result[n][0] == expected[n][0]
            assert result[n][1] == expected[n][1]
    else:
        assert result == expected

# check for unbalanced concatnation(add concat and 1x1 concat)
@pytest.mark.parametrize(("inputs", "expected"), [
    (99999, []), (1.0, [("conv2", "conv3"), ("conv1", "conv3")])
])
def test_judge_unbalanced_concatnation_add_conv1x1(inputs, expected):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0)

        def forward(self, x):
            x1 = self.conv1(x)
            x = torch.add(x, x1) 
            x = self.conv2(x)
            x = self.conv3(x)

            return x

    float_model = Model()

    def get_input_shape():
        return (3, 224, 224)
    
    tmpdir1 = tempfile.mkdtemp()
    repr_dataset = partial(random_data_gen, shape=get_input_shape())
    validation_dataset = partial(random_data_gen, use_labels=True)
    mct.set_log_folder(tmpdir1)
    mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                             representative_data_gen=repr_dataset)
   
    xquant_config = XQuantConfig(report_dir =  tmpdir1,
                                quantize_reported_dir = None,
                                threshold_ratio_unbalanced_concatenation = inputs
                                )
    # Initialize the logger with the report directory.
    result = judge_unbalanced_concatnation(degrade_layers=["conv1_add_collapsed", "conv2_conv3_collapsed"],
                                           float_model=float_model,
                                           dataset=repr_dataset,
                                           xquant_config=xquant_config
                                           )

    if(len(expected) > 0):
        for n in range(len(expected)):
            assert result[n][0] == expected[n][0]
            assert result[n][1] == expected[n][1]
    else:
        assert result == expected

# check for unbalanced concatnation(no set quantize_reported_dir)
@pytest.mark.parametrize(("inputs", "expected"), [
    (1.0, [])
])
def test_judge_unbalanced_concatnation_no_set_quantize_reported_dir(inputs, expected):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0)

        def forward(self, x):
            x1 = self.conv1(x)
            x = torch.add(x, x1) 
            x = self.conv2(x)
            x = self.conv3(x)

            return x

    float_model = Model()

    def get_input_shape():
        return (3, 224, 224)
    
    tmpdir1 = tempfile.mkdtemp()
    repr_dataset = partial(random_data_gen, shape=get_input_shape())
    validation_dataset = partial(random_data_gen, use_labels=True)
    Logger.shutdown()
    mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                             representative_data_gen=repr_dataset)
   
    xquant_config = XQuantConfig(report_dir =  tmpdir1,
                                quantize_reported_dir = None,
                                threshold_ratio_unbalanced_concatenation = inputs
                                )
    # Initialize the logger with the report directory.
    print(xquant_config.quantize_reported_dir)
    result = judge_unbalanced_concatnation(degrade_layers=["conv1_add_collapsed", "conv2_conv3_collapsed"],
                                           float_model=float_model,
                                           dataset=repr_dataset,
                                           xquant_config=xquant_config
                                           )

    if(len(expected) > 0):
        for n in range(len(expected)):
            assert result[n][0] == expected[n][0]
            assert result[n][1] == expected[n][1]
    else:
        assert result == expected

# check for unbalanced concatnation(set quantize_reported_dir but not exists data)
@pytest.mark.parametrize(("inputs", "expected"), [
    (1.0, [])
])
def test_judge_unbalanced_concatnation_no_exists_tensorboard_data(inputs, expected):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0)

        def forward(self, x):
            x1 = self.conv1(x)
            x = torch.add(x, x1) 
            x = self.conv2(x)
            x = self.conv3(x)

            return x

    float_model = Model()

    def get_input_shape():
        return (3, 224, 224)
    
    tmpdir1 = tempfile.mkdtemp()
    repr_dataset = partial(random_data_gen, shape=get_input_shape())
    validation_dataset = partial(random_data_gen, use_labels=True)
    mct.set_log_folder(tmpdir1)
    mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                             representative_data_gen=repr_dataset)
   
    xquant_config = XQuantConfig(report_dir =  tmpdir1,
                                quantize_reported_dir = None,
                                threshold_ratio_unbalanced_concatenation = inputs
                                )
    shutil.rmtree(xquant_config.quantize_reported_dir)
    # Initialize the logger with the report directory.
    print(xquant_config.quantize_reported_dir)
    result = judge_unbalanced_concatnation(degrade_layers=["conv1_add_collapsed", "conv2_conv3_collapsed"],
                                           float_model=float_model,
                                           dataset=repr_dataset,
                                           xquant_config=xquant_config
                                           )

    if(len(expected) > 0):
        for n in range(len(expected)):
            assert result[n][0] == expected[n][0]
            assert result[n][1] == expected[n][1]
    else:
        assert result == expected


# check for unbalanced concatnation(conv and bn concat)
@pytest.mark.parametrize(("inputs", "expected"), [
    (1.0, [("conv2", "bn")])
])
def test_judge_unbalanced_concatnation_conv_bn(inputs, expected):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.bn = torch.nn.BatchNorm2d(num_features=3)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.bn(x)
            return x

    float_model = Model()

    def get_input_shape():
        return (3, 224, 224)

    tmpdir1 = tempfile.mkdtemp()
    repr_dataset = partial(random_data_gen, shape=get_input_shape())
    validation_dataset = partial(random_data_gen, use_labels=True)
    mct.set_log_folder(tmpdir1)
    mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                                representative_data_gen=repr_dataset)

    xquant_config = XQuantConfig(report_dir =  tmpdir1,
                                quantize_reported_dir = None,
                                threshold_ratio_unbalanced_concatenation = inputs
                                )
    # Initialize the logger with the report directory.
    result = judge_unbalanced_concatnation(degrade_layers=["conv2_bn"],
                                            float_model=float_model,
                                            dataset=repr_dataset,
                                            xquant_config=xquant_config
                                            )
    
    if(len(expected) > 0):
        for n in range(len(expected)):
            assert result[n][0] == expected[n][0]
            assert result[n][1] == expected[n][1]
    else:
        assert result == expected


# check for unbalanced concatnation(bn and conv concat)
@pytest.mark.parametrize(("inputs", "expected"), [
    (1.0, [("bn", "conv2")])
])
def test_judge_unbalanced_concatnation_bn_conv(inputs, expected):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.pool = torch.nn.AvgPool2d(1, stride=1)
            self.bn = torch.nn.BatchNorm2d(num_features=3)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(x)
            x = self.bn(x)
            x = self.conv2(x)
            return x

    float_model = Model()

    def get_input_shape():
        return (3, 224, 224)

    tmpdir1 = tempfile.mkdtemp()
    repr_dataset = partial(random_data_gen, shape=get_input_shape())
    validation_dataset = partial(random_data_gen, use_labels=True)
    mct.set_log_folder(tmpdir1)
    mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                                representative_data_gen=repr_dataset)

    xquant_config = XQuantConfig(report_dir =  tmpdir1,
                                quantize_reported_dir = None,
                                threshold_ratio_unbalanced_concatenation = inputs
                                )
    # Initialize the logger with the report directory.
    result = judge_unbalanced_concatnation(degrade_layers=["bn_conv2"],
                                            float_model=float_model,
                                            dataset=repr_dataset,
                                            xquant_config=xquant_config
                                            )
    
    if(len(expected) > 0):
        for n in range(len(expected)):
            assert result[n][0] == expected[n][0]
            assert result[n][1] == expected[n][1]
    else:
        assert result == expected
