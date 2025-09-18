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
from typing import Dict
import os, shutil

import model_compression_toolkit as mct
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig


class TestXQuantConfig():

    def test_xquantconfig_not_report_dir(self):
        log_folder = "dummy_log_folder"
        mct.set_log_folder(log_folder)

        xquant_config = XQuantConfig(report_dir='dummy_report_dir', quantize_reported_dir=None)
        assert os.path.dirname(xquant_config.quantize_reported_dir) == log_folder

        shutil.rmtree(log_folder) # remove log_folder

    def test_xquantconfig_all(self):
        xquant_config = XQuantConfig(report_dir='dummy_report_dir', 
                                     quantize_reported_dir='dummy_quantized_reported_dir',
                                     threshold_quantize_error={"mse": 0.3, "cs": 0.3, "sqnr": 0.3},
                                     is_detect_under_threshold_quantize_error={"mse": True, "cs": False, "sqnr": False},
                                     threshold_degrade_layer_ratio=0.6,
                                     threshold_zscore_outlier_removal=3.0,
                                     threshold_ratio_unbalanced_concatenation=0.6,
                                     threshold_bitwidth_mixed_precision_with_model_output_loss_objective=4)
    
        assert xquant_config.report_dir == 'dummy_report_dir'
        assert xquant_config.quantize_reported_dir == 'dummy_quantized_reported_dir'

        assert isinstance(xquant_config.threshold_quantize_error, Dict)
        assert xquant_config.threshold_quantize_error["mse"] == 0.3
        assert xquant_config.threshold_quantize_error["cs"] == 0.3
        assert xquant_config.threshold_quantize_error["sqnr"] == 0.3

        assert isinstance(xquant_config.is_detect_under_threshold_quantize_error, Dict)
        assert xquant_config.is_detect_under_threshold_quantize_error["mse"] == True
        assert xquant_config.is_detect_under_threshold_quantize_error["cs"] == False
        assert xquant_config.is_detect_under_threshold_quantize_error["sqnr"] == False

        assert xquant_config.threshold_degrade_layer_ratio == 0.6
        assert xquant_config.threshold_zscore_outlier_removal == 3.0
        assert xquant_config.threshold_ratio_unbalanced_concatenation == 0.6
        assert xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective == 4
