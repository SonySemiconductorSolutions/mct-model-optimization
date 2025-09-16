#  Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
import pytest
import os
import glob
from functools import partial
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.graph_pb2 import GraphDef

import model_compression_toolkit as mct
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig
from model_compression_toolkit.xquant.pytorch.facade_xquant_report import xquant_report_troubleshoot_pytorch_experimental
from model_compression_toolkit.xquant.common.similarity_functions import DEFAULT_SIMILARITY_METRICS_NAMES
from model_compression_toolkit.xquant.common.constants import OUTPUT_SIMILARITY_METRICS_REPR, \
    OUTPUT_SIMILARITY_METRICS_VAL, INTERMEDIATE_SIMILARITY_METRICS_REPR, INTERMEDIATE_SIMILARITY_METRICS_VAL, \
    XQUANT_REPR, XQUANT_VAL, CUT_MEMORY_ELEMENTS, CUT_TOTAL_SIZE
from mct_quantizers import PytorchQuantizationWrapper

from tests.pytorch_tests.xquant_tests.test_xquant_end2end import random_data_gen
from tests.common_tests.helpers.tpcs_for_tests.v2.tpc import get_tpc


def random_data_gen(shape=(3, 8, 8), use_labels=False, num_inputs=1, batch_size=2, num_iter=2):
    if use_labels:
        for _ in range(num_iter):
            yield [[torch.randn(batch_size, *shape)] * num_inputs, torch.randn(batch_size)]
    else:
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

class TestXQuantReportModel1:
    def get_input_shape(self):
        return (3, 8, 8)

    def get_core_config(self):
        return mct.core.CoreConfig(debug_config=mct.core.DebugConfig(simulate_scheduler=True))

    def get_tpc(self):
        return get_tpc()

    def get_model_to_test(self):
        class BaseModelTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.bn = torch.nn.BatchNorm2d(num_features=3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x
        return BaseModelTest()
    
    def setup_environment(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ptq_tb_dir = os.path.join(self.tmpdir, "ptq_tb_dir")
        mct.set_log_folder(self.ptq_tb_dir)

        self.float_model = self.get_model_to_test()
        self.repr_dataset = partial(random_data_gen, shape=self.get_input_shape())
        self.quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=self.float_model,
                                                                                representative_data_gen=self.repr_dataset,
                                                                                target_platform_capabilities=get_tpc(),
                                                                                core_config=self.get_core_config())
        self.validation_dataset = partial(random_data_gen, use_labels=True)
        self.xquant_config = XQuantConfig(report_dir=self.tmpdir, quantize_reported_dir=self.ptq_tb_dir)

    def clean_environment(self):
        del self.float_model
        del self.quantized_model
        del self.repr_dataset
        del self.validation_dataset
        torch.cuda.empty_cache()

    def test_xquant_report_output_metrics(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert OUTPUT_SIMILARITY_METRICS_VAL in result_quant
        assert OUTPUT_SIMILARITY_METRICS_REPR in result_quant
        assert len(result_quant[OUTPUT_SIMILARITY_METRICS_REPR])==len(DEFAULT_SIMILARITY_METRICS_NAMES)
        assert len(result_quant[OUTPUT_SIMILARITY_METRICS_VAL])==len(DEFAULT_SIMILARITY_METRICS_NAMES)
        self.clean_environment()

    def test_intermediate_metrics(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        assert INTERMEDIATE_SIMILARITY_METRICS_REPR in result_quant
        linear_layers = [n for n,m in self.quantized_model.named_modules() if isinstance(m, PytorchQuantizationWrapper)]

        assert linear_layers[0] in result_quant[INTERMEDIATE_SIMILARITY_METRICS_REPR]
        for k,v in result_quant[INTERMEDIATE_SIMILARITY_METRICS_REPR].items():
            assert len(v)==len(DEFAULT_SIMILARITY_METRICS_NAMES)

        assert INTERMEDIATE_SIMILARITY_METRICS_VAL in result_quant
        for k,v in result_quant[INTERMEDIATE_SIMILARITY_METRICS_VAL].items():
            assert len(v)==len(DEFAULT_SIMILARITY_METRICS_NAMES)
        self.clean_environment()

    def test_custom_metric(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}
        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        assert OUTPUT_SIMILARITY_METRICS_REPR in result_quant
        assert len(result_quant[OUTPUT_SIMILARITY_METRICS_REPR])==len(DEFAULT_SIMILARITY_METRICS_NAMES) + 1
        assert "mae" in result_quant[OUTPUT_SIMILARITY_METRICS_REPR]
        assert INTERMEDIATE_SIMILARITY_METRICS_REPR in result_quant

        for k,v in result_quant[INTERMEDIATE_SIMILARITY_METRICS_REPR].items():
            assert len(v)==len(DEFAULT_SIMILARITY_METRICS_NAMES) + 1
            assert "mae" in v
        self.clean_environment()

    def test_tensorboard_graph(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        events_dir = os.path.join(self.xquant_config.report_dir, 'xquant')
        initial_graph_events_files = glob.glob(events_dir + '/*events*')
        initial_graph_event = initial_graph_events_files[0]
        efl = event_file_loader.LegacyEventFileLoader(initial_graph_event).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        for node in g.node:
            if node.device == 'PytorchQuantizationWrapper':
                assert XQUANT_REPR in str(node)
                assert XQUANT_VAL in str(node)
                assert CUT_MEMORY_ELEMENTS in str(node)
                assert CUT_TOTAL_SIZE in str(node)
        self.clean_environment()

    def test_troubleshoot(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()


    def test_troubleshoot_not_pass_threshold_quantize_error(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 99999, "cs": -1, "sqnr": -1}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()


    def test_troubleshoot_not_pass_threshold_degrade_layer_ratio(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=0.0
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" not in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True
        

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert os.path.exists(os.path.join(self.tmpdir, "outlier_histgrams"))==False
        self.clean_environment()


    def test_troubleshoot_not_pass_threshold_judge_troubleshoots(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=99999
        self.xquant_config.threshold_ratio_unbalanced_concatenation=99999
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=1


        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" not in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()


# Test with Conv2D without BatchNormalization and without Activation
class TestXQuantReportModel2(TestXQuantReportModel1):

    def get_model_to_test(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, x):
                x1 = self.conv1(x)
                x = x + x1
                x = F.softmax(x, dim=1)
                return x

        return Model()
    
    def test_troubleshoot(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))!=0
        self.clean_environment()

    def test_troubleshoot_not_pass_threshold_quantize_error(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 99999, "cs": -1, "sqnr": -1}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()
    

# Test with Multiple Convolution Layers and an Addition Operator
class TestXQuantReportModel3(TestXQuantReportModel1):
    def get_model_to_test(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x = x1 + x2
                x = self.conv3(x)
                x = F.softmax(x, dim=1)
                return x

        return Model()
    
    def test_troubleshoot(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))!=0
        self.clean_environment()

    def test_troubleshoot_not_pass_threshold_quantize_error(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 99999, "cs": -1, "sqnr": -1}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()


# Test with Conv2D and a Hardswish activation(for shift negative activation)
class TestXQuantReportModel4(TestXQuantReportModel1):

    def get_model_to_test(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.activation = nn.Hardswish()

            def forward(self, x):
                x = self.conv1(x)
                x = self.activation(x)
                x = F.softmax(x, dim=1)
                return x

        return Model()

    def test_troubleshoot(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        #assert "outlier_removal" in result_troubleshoot # Not detected from one layer model.
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert os.path.exists(os.path.join(self.tmpdir, "outlier_histgrams"))==True
        self.clean_environment()
    
    def test_troubleshoot_not_pass_threshold_quantize_error(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 99999, "cs": -1, "sqnr": -1}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()

    def test_troubleshoot_not_pass_threshold_judge_troubleshoots(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=99999
        self.xquant_config.threshold_ratio_unbalanced_concatenation=99999
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=1


        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" not in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True
        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()


# Test with Conv2D and a 1x1 Conv2D(for unbalanced concatenation)
class TestXQuantReportModel5(TestXQuantReportModel1):
    def get_model_to_test(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(3, 3, kernel_size=1, padding=0)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = F.softmax(x, dim=1)
                return x

        return Model()

    def test_troubleshoot(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999    

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True
        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))!=0
        self.clean_environment()

    def test_troubleshoot_not_pass_threshold_quantize_error(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 99999, "cs": -1, "sqnr": -1}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()


class TestXQuantReportModelMBv2(TestXQuantReportModel1):

    def get_input_shape(self):
        return (3, 224, 224)

    def get_model_to_test(self):
        from torchvision.models.mobilenetv2 import MobileNetV2
        return MobileNetV2()
    
    def test_troubleshoot(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999    

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert os.path.exists(os.path.join(self.tmpdir, "outlier_histgrams"))==True
        self.clean_environment()

    def test_troubleshoot_not_pass_threshold_quantize_error(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 99999, "cs": -1, "sqnr": -1}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()


class TestXQuantReportModelMBv3(TestXQuantReportModel1):

    def get_input_shape(self):
        return (3, 224, 224)

    def get_model_to_test(self):
        from torchvision.models.mobilenet import mobilenet_v3_small
        return mobilenet_v3_small()

    def test_troubleshoot(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" in result_troubleshoot
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True
        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))!=0
        self.clean_environment()

    def test_troubleshoot_not_pass_threshold_judge_troubleshoots(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=99999
        self.xquant_config.threshold_ratio_unbalanced_concatenation=99999
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999


        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True
        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()

    def test_troubleshoot_not_pass_threshold_quantize_error(self):
        self.setup_environment()
        self.xquant_config.custom_similarity_metrics = None
        self.xquant_config.threshold_quantize_error = {"mse": 99999, "cs": -1, "sqnr": -1}
        self.xquant_config.threshold_degrade_layer_ratio=1.1
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999

        result_quant, result_troubleshoot = xquant_report_troubleshoot_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

        assert os.path.exists(os.path.join(self.tmpdir, "troubleshoot_report.json"))==True

        for metrics_name in DEFAULT_SIMILARITY_METRICS_NAMES:
            for dataset_name in ["repr", "val"]:
                assert os.path.exists(os.path.join(self.tmpdir, "quant_loss_{}_{}.png".format(metrics_name, dataset_name)))==True

        assert len(os.listdir(os.path.join(self.tmpdir, "outlier_histgrams")))==0
        self.clean_environment()
