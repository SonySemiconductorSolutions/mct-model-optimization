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

import os
from typing import Callable, Any, Dict
from model_compression_toolkit.xquant import XQuantConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.xquant.pytorch.judge_troubleshoot_utils import judge_outlier_removal,  judge_shift_negative_activation, judge_unbalanced_concatnation, judge_mixed_precision_with_model_output_loss_objective

def core_judge_troubleshoot(float_model: Any,
                            quantized_model: Any,
                            float_graph: Graph,
                            degrade_layers: list[str],
                            dataset: Callable,
                            xquant_config: XQuantConfig) -> Dict[str, Any]:
    """
    Judge whether judgeable troubleshoots and make troubleshoot report.

    Args:
        float_model (Any): The original floating-point model.
        quantized_model (Any): The model after quantization.
        float_graph (Graph): Graph of float_model with histgrams.
        degrade_layers (list[str]): A list of detected degrade layers.
        dataset (Callable): Representative dataset used for similarity metrics computation.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        Dict[str, Any]: A dictionary containing the analyze degrade cause report for degraded layers.
    """

    _troubleshoot_data = {"outlier_removal":[], "shift_negative_activation":[], "unbalanced_concatenation":[], "mixed_precision_with_model_output_loss_objective":[]}

    # Judge whether the layer has outliers from statistics information
    # make outlier image folder
    outlier_histgram_dir = os.path.join(xquant_config.report_dir, "outlier_histgrams")
    if(not os.path.exists(outlier_histgram_dir)):
        os.mkdir(outlier_histgram_dir)
    _troubleshoot_data["outlier_removal"] = judge_outlier_removal(degrade_layers, float_graph, xquant_config)

    # Judge whether the layer combines layers with significantly different value ranges
    _troubleshoot_data["unbalanced_concatenation"] = judge_unbalanced_concatnation(degrade_layers, float_model, dataset, xquant_config)

    # Judge whether the layer has a negative activation function (PReLU / ELU / Hardswish / SiLU / GELU)
    _troubleshoot_data["shift_negative_activation"] = judge_shift_negative_activation(float_graph, xquant_config)

    # Judge whether the bitwidth of the final layer is less than threshold
    _troubleshoot_data["mixed_precision_with_model_output_loss_objective"] = judge_mixed_precision_with_model_output_loss_objective(quantized_model, xquant_config)

    # Delete no data key
    for troubleshoot_name in list(_troubleshoot_data.keys()):
        if(len(_troubleshoot_data[troubleshoot_name]) == 0):
            del _troubleshoot_data[troubleshoot_name]

    return _troubleshoot_data