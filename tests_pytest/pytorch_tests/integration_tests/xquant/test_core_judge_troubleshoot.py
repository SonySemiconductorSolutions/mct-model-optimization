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
from functools import partial
import tempfile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_compression_toolkit as mct
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig
from model_compression_toolkit.xquant.pytorch.pytorch_report_utils import PytorchReportUtils
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.xquant.pytorch.core_judge_troubleshoot import core_judge_troubleshoot
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

    def setup_environment(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ptq_tb_dir = os.path.join(self.tmpdir, "ptq_tb_dir")
        mct.set_log_folder(self.ptq_tb_dir)

        self.repr_dataset = partial(random_data_gen, shape=self.get_input_shape())
        self.validation_dataset = partial(random_data_gen, use_labels=True)
        self.xquant_config = XQuantConfig(report_dir=self.tmpdir, quantize_reported_dir=self.ptq_tb_dir)

    def setup_exec_before_core_judge_troubleshoot(self):
        # exec PTQ
        self.quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=self.float_model,
                                                                             representative_data_gen=self.repr_dataset,
                                                                             target_platform_capabilities=get_tpc(),
                                                                             core_config=self.get_core_config()
                                                                            )
        # exec xquant(before core_judge_troubleshoot)
        self.pytorch_report_utils = PytorchReportUtils(self.xquant_config.report_dir)
        self.float_graph = self.pytorch_report_utils.model_folding_utils.create_float_folded_graph(self.float_model, self.repr_dataset)

        mi = ModelCollector(self.float_graph, self.pytorch_report_utils.fw_impl, self.pytorch_report_utils.fw_info)
        for _data in tqdm(self.repr_dataset(), desc="Collecting Histograms"):
            mi.infer(_data)

    def test_all_detection(self):
        self.setup_environment()
        def get_model():
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.bn = torch.nn.BatchNorm2d(num_features=3)
                    self.prelu = nn.PReLU()

                def forward(self, x):
                    x1 = self.conv1(x)
                    x2 = self.conv2(x)
                    x = x1 + x2
                    x = self.conv3(x)
                    x = self.bn(x)
                    x = self.prelu(x)
                    x = F.softmax(x, dim=1)
                    return x

            return Model()
        self.float_model = get_model()
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999
        self.degrade_layers = ["conv1","conv2","conv3_bn"]
        self.setup_exec_before_core_judge_troubleshoot()

        result_troubleshoot = core_judge_troubleshoot(self.float_model,
                                            self.quantized_model,
                                            self.float_graph,
                                            self.degrade_layers,
                                            self.validation_dataset,
                                            self.xquant_config)

        assert "outlier_removal" in result_troubleshoot
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot
    
    def test_outlier_removal_detection(self):
        self.setup_environment()
        def get_model():
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
        self.float_model = get_model()
        self.xquant_config.threshold_zscore_outlier_removal=0.1
        self.xquant_config.threshold_ratio_unbalanced_concatenation=99999
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=2
        self.degrade_layers = ["conv1","conv2","conv3"]
        self.setup_exec_before_core_judge_troubleshoot()

        result_troubleshoot = core_judge_troubleshoot(self.float_model,
                                            self.quantized_model,
                                            self.float_graph,
                                            self.degrade_layers,
                                            self.validation_dataset,
                                            self.xquant_config)

        assert "outlier_removal" in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" not in result_troubleshoot
    
    def test_unbalanced_concatnation_detection(self):
        self.setup_environment()
        def get_model():
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.bn = torch.nn.BatchNorm2d(num_features=3)

                def forward(self, x):
                    x1 = self.conv1(x)
                    x2 = self.conv2(x)
                    x = x1 + x2
                    x = self.conv3(x)
                    x = self.bn(x)
                    x = F.softmax(x, dim=1)
                    return x

            return Model()
        self.float_model = get_model()
        self.xquant_config.threshold_zscore_outlier_removal=99999
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=2
        self.degrade_layers = ["conv1","conv2","conv3_bn"]
        self.setup_exec_before_core_judge_troubleshoot()

        result_troubleshoot = core_judge_troubleshoot(self.float_model,
                                            self.quantized_model,
                                            self.float_graph,
                                            self.degrade_layers,
                                            self.validation_dataset,
                                            self.xquant_config)

        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" not in result_troubleshoot
    
    def test_shift_negative_activation_detection(self):
        def get_model():
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                    self.bn = torch.nn.BatchNorm2d(num_features=3)
                    self.prelu = nn.PReLU()

                def forward(self, x):
                    x1 = self.conv1(x)
                    x2 = self.conv2(x)
                    x = x1 + x2
                    x = self.conv3(x)
                    x = self.bn(x)
                    x = self.prelu(x)
                    x = F.softmax(x, dim=1)
                    return x

            return Model()
        self.float_model = get_model()

        self.setup_environment()
        self.xquant_config.threshold_zscore_outlier_removal=99999
        self.xquant_config.threshold_ratio_unbalanced_concatenation=99999
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=2
        self.degrade_layers = ["conv1","conv2","conv3_bn"]
        self.setup_exec_before_core_judge_troubleshoot()

        result_troubleshoot = core_judge_troubleshoot(self.float_model,
                                            self.quantized_model,
                                            self.float_graph,
                                            self.degrade_layers,
                                            self.validation_dataset,
                                            self.xquant_config)

        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" not in result_troubleshoot

    def test_mixed_precision_detection(self):
        self.setup_environment()
        def get_model():
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
        self.float_model = get_model()
        self.xquant_config.threshold_zscore_outlier_removal=99999
        self.xquant_config.threshold_ratio_unbalanced_concatenation=99999
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=99999
        self.degrade_layers = ["conv1","conv2","conv3"]
        self.setup_exec_before_core_judge_troubleshoot()

        result_troubleshoot = core_judge_troubleshoot(self.float_model,
                                            self.quantized_model,
                                            self.float_graph,
                                            self.degrade_layers,
                                            self.validation_dataset,
                                            self.xquant_config)

        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" in result_troubleshoot

    def test_all_notdetection(self):
        self.setup_environment()
        def get_model():
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
        self.float_model = get_model()
        self.xquant_config.threshold_zscore_outlier_removal=99999
        self.xquant_config.threshold_ratio_unbalanced_concatenation=1.0
        self.xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective=2
        self.degrade_layers = ["conv1","conv2","conv3"]
        self.setup_exec_before_core_judge_troubleshoot()  

        result_troubleshoot = core_judge_troubleshoot(self.float_model,
                                            self.quantized_model,
                                            self.float_graph,
                                            self.degrade_layers,
                                            self.validation_dataset,
                                            self.xquant_config)

        assert "outlier_removal" not in result_troubleshoot
        assert "shift_negative_activation" not in result_troubleshoot
        assert "unbalanced_concatenation" not in result_troubleshoot
        assert "mixed_precision_with_model_output_loss_objective" not in result_troubleshoot
