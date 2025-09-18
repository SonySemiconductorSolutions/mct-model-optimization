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
from functools import partial
import os
import tempfile

import torch
import torch.nn as nn

import model_compression_toolkit as mct
from model_compression_toolkit.xquant.pytorch.core_report_generator import core_report_generator_troubleshoot
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig
from model_compression_toolkit.xquant.pytorch.pytorch_report_utils import PytorchReportUtils


def random_data_gen(shape=(3, 8, 8), use_labels=False, num_inputs=1, batch_size=2, num_iter=2):
    if use_labels:
        for _ in range(num_iter):
            yield [[torch.randn(batch_size, *shape)] * num_inputs, torch.randn(batch_size)]
    else:
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

def get_input_shape():
        return (3, 8, 8)

def get_core_config():
    return mct.core.CoreConfig(debug_config=mct.core.DebugConfig(simulate_scheduler=True))

def get_tpc():
    return get_tpc()

def get_model_to_test():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
            self.activation = nn.Hardswish()

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x = x1 + x2
            x = self.conv3(x)
            x = self.activation(x)
            return x

    return Model()

def setup_environment():
        tmpdir = tempfile.mkdtemp()
        ptq_tb_dir = os.path.join(tmpdir, "ptq_tb_dir")
        mct.set_log_folder(ptq_tb_dir)

        float_model = get_model_to_test()
        repr_dataset = partial(random_data_gen, shape=get_input_shape())
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                                                                representative_data_gen=repr_dataset,)
        validation_dataset = partial(random_data_gen, use_labels=True)
        xquant_config = XQuantConfig(report_dir=tmpdir, quantize_reported_dir=ptq_tb_dir,threshold_quantize_error = {"mse": 0.0, "cs": 0.0, "sqnr": 0.0})

        return tmpdir, ptq_tb_dir, float_model, repr_dataset, quantized_model, validation_dataset, xquant_config

def test_degrade_layer_detect():
    tmpdir, ptq_tb_dir, float_model, repr_dataset, quantized_model, validation_dataset, xquant_config = setup_environment()
    xquant_config.threshold_degrade_layer_ratio = 1.1
    pytorch_report_utils = PytorchReportUtils(xquant_config.report_dir)
    result1, result2 = core_report_generator_troubleshoot(float_model,
                                    quantized_model,
                                    repr_dataset,
                                    validation_dataset,
                                    pytorch_report_utils,
                                    xquant_config)
    assert len(result1) != 0
    assert len(result2) != 0

def test_degrade_layer_not_detect():
    tmpdir, ptq_tb_dir, float_model, repr_dataset, quantized_model, validation_dataset, xquant_config = setup_environment()
    xquant_config.threshold_degrade_layer_ratio = 0.0
    pytorch_report_utils = PytorchReportUtils(xquant_config.report_dir)
    result1, result2 = core_report_generator_troubleshoot(float_model,
                                    quantized_model,
                                    repr_dataset,
                                    validation_dataset,
                                    pytorch_report_utils,
                                    xquant_config)
    assert len(result1) != 0
    assert len(result2) == 0
    
     
