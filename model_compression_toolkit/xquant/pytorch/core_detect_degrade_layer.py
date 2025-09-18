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

from typing import Callable, Any, Dict
from model_compression_toolkit.xquant import XQuantConfig
from model_compression_toolkit.xquant.pytorch.detect_degrade_utils import make_similarity_graph
from model_compression_toolkit.logger import Logger

def core_detect_degrade_layer(repl_similarity: tuple[Dict[str, Any], Dict[str, Any]],
                              val_similarity: tuple[Dict[str, Any], Dict[str, Any]],
                              xquant_config: XQuantConfig) -> list[str]:
    """
    Detect degrade layers by caliculated similarities by XQuant.
    And Draw and save similarity graphs.

    Args:
        repr_similarity (tuple[Dict[str, Any], Dict[str, Any]]): Quant error reports of Representative dataset.
        val_similarity (tuple[Dict[str, Any], Dict[str, Any]]): Quant error reports of Validation dataset.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        List[str]: A list of detected degrade layers.
    """

    degrade_layers = []
    metrics_names = repl_similarity[0].keys()
    for metrics_name in metrics_names:
        # Check xquant_config parameter of custom_similarity_metrics (If not threshold then skip. If not flag of under/above threshold detection then assume above threshold detection.)
        if(metrics_name not in xquant_config.threshold_quantize_error.keys()):
            Logger.warning("XQuantConfig.threshold_quantize_error[\'{}\'] is not defined. Skipping detection degrade layers by \'{}\'".format(metrics_name, metrics_name))
            continue
        if(metrics_name not in xquant_config.is_detect_under_threshold_quantize_error.keys()):
            Logger.warning("XQuantConfig.is_detect_under_threshold_quantize_error[{}] is not defined. Assume =False".format(metrics_name))
            xquant_config.is_detect_under_threshold_quantize_error[metrics_name] = False

        for dataset_similarity, dataset_name in [(repl_similarity, "repr"), (val_similarity, "val")]:
            degrade_layers_tmp = []
            intermediate_similarity = dataset_similarity[1]
            for layer_name in intermediate_similarity.keys():
                quantize_error = intermediate_similarity[layer_name][metrics_name]
                threshold_quantize_error = xquant_config.threshold_quantize_error[metrics_name]
                
                # Switch by under/above threshold flag of xquant_config
                is_degrade = False
                if(xquant_config.is_detect_under_threshold_quantize_error[metrics_name]):
                    if(quantize_error <= threshold_quantize_error):
                        is_degrade = True        
                else:
                    if(quantize_error >= threshold_quantize_error):
                        is_degrade = True  

                # Add degrade_layers
                if(is_degrade):
                    # Print to Console
                    if(len(degrade_layers)==0):
                        Logger.info("This may be problematic because the quantization error is larger than other layers. Refer to the TroubleShooting Documentation (MCT XQuant Extension Tool).")
                    Logger.info("{}[{}]={}".format(metrics_name, layer_name, quantize_error))
                    if(layer_name not in degrade_layers_tmp):
                        degrade_layers_tmp.append(layer_name)
                    if(layer_name not in degrade_layers):
                        degrade_layers.append(layer_name)
            # Draw and save similarity graph by matplotlib
            make_similarity_graph(metrics_name, dataset_name, intermediate_similarity, degrade_layers_tmp, xquant_config)
                    
    return degrade_layers