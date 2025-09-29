#  Copyright 2024 Sony Semiconductor Solutions, Inc. All rights reserved.
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

from typing import Dict, Callable
from model_compression_toolkit.logger import Logger


class XQuantConfig:
    """
    Configuration for generating the report.
    It allows to set the log dir that the report will be saved in and to add similarity metrics
    to measure between tensors of the two models.
    """

    def __init__(self,
                 report_dir: str,
                 custom_similarity_metrics: Dict[str, Callable] = None,
                 quantize_reported_dir: str = None,
                 threshold_quantize_error: Dict[str, float] = {"mse": 0.1, "cs": 0.1, "sqnr": 0.1},
                 is_detect_under_threshold_quantize_error: Dict[str, bool] = {"mse": False, "cs": True, "sqnr": True},
                 threshold_degrade_layer_ratio: float = 0.5,
                 threshold_zscore_outlier_removal: float = 5.0,
                 threshold_ratio_unbalanced_concatenation: float = 16.0,
                 threshold_bitwidth_mixed_precision_with_model_output_loss_objective: int = 2
                 ):
        """
        Initializes the configuration for explainable quantization.

        Args:
            report_dir (str): Directory where the reports will be saved.
            custom_similarity_metrics (Dict[str, Callable]): Custom similarity metrics to be computed between tensors of the two models. The dictionary keys are similarity metric names and the values are callables that implement the similarity metric computation.
            quantize_reported_dir (str): Directory where the the quantization log will be saved. 
            threshold_quantize_error (Dict[str, float]): Threshold values for detecting degradation in accuracy.
            is_detect_under_threshold_quantize_error (Dict[str, bool]): For each threshold specified in threshold_quantize_error, True: detect the layer as degraded when the error is below the threshold.; False: detect the layer as degraded when the error is above the threshold.
            threshold_degrade_layer_ratio (float): If the number of layers detected as degraded is large, skips the judge degradation causes Specify the ratio here.
            threshold_zscore_outlier_removal (float): Used in judge degradation causes (Outlier Removal). Threshold for z_score to detect outliers.
            threshold_ratio_unbalanced_concatenation (float): Used in judge degradation causes (unbalanced “concatnation”). Threshold for the multiplier of range width between concatenated layers.
            threshold_bitwidth_mixed_precision_with_model_output_loss_objective (int): Used in judge degradation causes (Mixed precision with model output loss objective). Bitwidth of the final layer to judge insufficient bitwidth.
        """

        self.report_dir = report_dir
        self.custom_similarity_metrics = custom_similarity_metrics
        self.quantize_reported_dir = quantize_reported_dir
        if(self.quantize_reported_dir is None):
            self.quantize_reported_dir = Logger.LOG_PATH
        self.threshold_quantize_error = threshold_quantize_error
        self.is_detect_under_threshold_quantize_error = is_detect_under_threshold_quantize_error
        self.threshold_degrade_layer_ratio = threshold_degrade_layer_ratio
        self.threshold_zscore_outlier_removal = threshold_zscore_outlier_removal
        self.threshold_ratio_unbalanced_concatenation = threshold_ratio_unbalanced_concatenation
        self.threshold_bitwidth_mixed_precision_with_model_output_loss_objective = threshold_bitwidth_mixed_precision_with_model_output_loss_objective