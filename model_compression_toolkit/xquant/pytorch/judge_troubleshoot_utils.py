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
from functools import partial
from typing import Tuple, List, Callable
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from model_compression_toolkit.xquant import XQuantConfig
from model_compression_toolkit.xquant.pytorch.dataset_utils import PytorchDatasetUtils
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
import torch
from torch.nn import Hardswish, SiLU, PReLU, ELU, GELU
from torch.nn.functional import hardswish, silu, prelu, elu, gelu
from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator
from model_compression_toolkit.logger import Logger


def _compute_zscore(statistics_collector: BaseStatsCollector) -> Tuple[np.array, np.array, np.array]:
    """
    Compute z-score from collected histogram.

    Args:
        statistics_collector (BaseStatsCollector): Statistics collector to compute z-score.

    Returns:
        Tuple[np.array, np.array, np.array]: A tuple containing computed z-score, histogram statistics (bins and counts).
    """

    if statistics_collector.require_collection():
        if hasattr(statistics_collector, 'hc'):
            if statistics_collector.hc.is_legal:
                bins, counts = statistics_collector.hc.get_histogram()
                if bins is not None and counts is not None:
                    bins = np.copy(bins)
                    counts = np.copy(counts)
                    bins = bins[:-1]  # take out the last range

                    # Compute the z-score
                    mu = np.sum(bins * counts) / np.sum(counts)
                    sigma = np.sqrt(np.sum(np.power(bins - mu, 2.0) * counts) / np.sum(counts))
                    z_score = np.abs(bins - mu) / sigma

                    return (z_score, bins, counts, mu, sigma)
    
    return None


def _save_outlier_histogram(layer_name: str, zscore_hist: Tuple[np.ndarray, np.ndarray, np.ndarray, float, float], 
                            z_threshold: float, img_filename: str):
    """
    Save output activation distributions histogram.

    Args:
        layer_name (str): The name of the layer.
        zscore_hist (Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]): A tuple containing z-score, histogram statistics.
        z_threshold (float): Threshold for detecting outliers.
        img_filename (str): Filename to save histogram image to.  
    """
    zscore, bins, counts, mu, sigma = zscore_hist
    #zscore_over = zscore >= z_threshold

    # Compute bin centers
    bin_centers = bins

    # Thresholds
    lower_threshold = -z_threshold
    upper_threshold = z_threshold
    #z_score_vline = z_threshold * sigma + mu
    """ if bins-mu >= 0:
        z_score_vline = mu+z_threshold*(sigma+10e-16)
    else:
        z_score_vline = mu-z_threshold*(sigma+10e-16) """

    lower_th_z_score_vline = mu-z_threshold*(sigma+10e-16)
    upper_th_z_score_vline = mu+z_threshold*(sigma+10e-16)


    def detect_outliers_after_peak(counts, bin_centers, lower_thresh, upper_thresh):
        """
        Filtering outlier data in the histogram that are outside the threshold and not monotonically decreasing.

        Args:
            counts (np.ndarray): z-score histogram counts.
            bin_centers (np.ndarray): the centor data of histogram counts.
            lower_thresh (float): Threshold for detecting outliers(lower side).
            upper_thresh (float): Threshold for detecting outliers(upper side).
        Returns:
            color (np.ndarray): color data of filterd histogram.
        """

        # Detect outliers: once a bin exceeds the threshold and is greater than its neighbor,
        # all subsequent bins are considered outliers.
        
        colors = ['blue'] * len(counts)
        upper_outlier_indices = []
        lower_outlier_indices = []


        # Check for positive side (right-facing peak)
        for i in range(len(counts)):
            if bin_centers[i] > upper_thresh:
                if i > 0 and counts[i] > counts[i - 1]:
                    upper_outlier_indices = list(range(i, len(counts)))
                    break

        # Check for negative side (left-facing peak)
        for i in reversed(range(len(counts))):
            if bin_centers[i] < lower_thresh:
                if i < len(counts) - 1 and counts[i] > counts[i + 1]:
                    lower_outlier_indices = list(range(0, i + 1))
                    break

        for idx1 in upper_outlier_indices:
            colors[idx1] = 'red'
        
        for idx2 in lower_outlier_indices:
            colors[idx2] = 'red'

        return colors


    # Detect outliers
    colors = detect_outliers_after_peak(counts, bin_centers, lower_th_z_score_vline, upper_th_z_score_vline)

    # Plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax_top = ax.twiny()

    ax.bar(bin_centers, counts, width=0.9 * (bins[1] - bins[0]), color=colors)
    ax.set_ylabel('counts')
    ax.set_xlabel('bins')

    bin_ticks = ax.get_xticks()
    z_score_ticks = np.abs(bin_ticks - mu) / (sigma + 1e-16)

    ax.set_xticks(bin_ticks)
    ax_top.set_xticks(bin_ticks)
    ax_top.set_xticklabels([f'{zs:.2f}' for zs in z_score_ticks])
    ax_top.set_xlabel('z-score')
    plt.ylim(0, np.max(counts) * 1.1)
    ax.set_title(layer_name)

    xlim_min, xlim_max = ax.get_xlim()
    xlim_range = xlim_max - xlim_min

    upper_zscore=abs(max(bins) - mu) / (sigma + 1e-16)
    ax.axvline(x=max(bins), color='black', linestyle='--')
    plt.text(max(bins) - 0.18*xlim_range, plt.ylim()[1] * 0.9, f'upper zscore={upper_zscore:.2f}', color='black')

    lower_zscore=abs(min(bins) - mu) / (sigma + 1e-16)
    ax.axvline(x=min(bins), color='black', linestyle='--')
    plt.text(min(bins) + 0.01*xlim_range, plt.ylim()[1] * 0.9, f'lower zscore={lower_zscore:.2f}', color='black')

    if(upper_th_z_score_vline <= xlim_max):
        ax_top.axvline(x=upper_th_z_score_vline, color='r', linestyle='--')
        plt.text(upper_th_z_score_vline - 0.12*xlim_range, plt.ylim()[1] * 0.95, f'threshold={z_threshold}', color='red')
    else:
        plt.text(xlim_max - 0.18*xlim_range, plt.ylim()[1] * 0.95, f'threshold={z_threshold} ->', color='red')
    
    if(xlim_min <= lower_th_z_score_vline):
        ax_top.axvline(x=lower_th_z_score_vline, color='r', linestyle='--')
        plt.text(lower_th_z_score_vline + 0.01*xlim_range, plt.ylim()[1] * 0.95, f'threshold={z_threshold}', color='red')
    else:
        plt.text(xlim_min + 0.01*xlim_range, plt.ylim()[1] * 0.95, f'<- threshold={z_threshold}', color='red')


    plt.tight_layout()

    plt.savefig(img_filename)
    plt.clf()
    plt.close()


def judge_outlier_removal(degrade_layers: list[str], float_graph: Graph,
                          xquant_config: XQuantConfig) -> Tuple[str, str]:
    """
    Judge whether the degrade layers have outliers from statistics information

    Args:
        degrade_layers (list[str]): A list of detected degrade layers.
        float_graph (Graph): Graph to get statistics for the layer. 
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        Tuple[str, str]: A tuple containing the layer name with outliers and the image filename saved the histogram.
                         If the layer does not match the condition, it returns None.
    """
    z_threshold = xquant_config.threshold_zscore_outlier_removal

    outlier_layers = []

    for layer_name in degrade_layers:
        nodes = float_graph.find_node_by_name(layer_name)
        assert len(nodes) == 1  # check length of nodes

        collector = float_graph.get_out_stats_collector(nodes[0])
        if collector is not None:
            statistics = float_graph.get_out_stats_collector(nodes[0])
            zscore_hist = _compute_zscore(statistics)

            if zscore_hist is not None:
                zscore = zscore_hist[0]

                if np.any(zscore >= z_threshold):
                    img_filename = os.path.join(xquant_config.report_dir, f'outlier_histgrams', f'{layer_name}.png')
                    if(os.path.exists(os.path.join(xquant_config.report_dir, f'outlier_histgrams'))):
                        _save_outlier_histogram(layer_name, zscore_hist, z_threshold, img_filename)
                    else:
                        Logger.warning("Output directory of outlier histgram images({}/outlier_histgrams) not found. Skipping output outlier histgram images.".format(xquant_config.report_dir))

                    # Print to Console
                    if(len(outlier_layers) == 0):
                        Logger.warning("There are output values ​​that deviate significantly from the average. Refer to the following images and the TroubleShooting Documentation (MCT XQuant Extension Tool) of \'Outlier Removal\'.")
                    Logger.warning(img_filename)

                    outlier_layers.append((layer_name, img_filename))

    return outlier_layers


def judge_shift_negative_activation(float_graph: Graph,
                                    xquant_config: XQuantConfig) -> list[str]:
    """
    Judge whether the layer has a negative activation function (PReLU / ELU / Hardswish / SiLU / GELU)

    Args:
        float_graph (Graph): Graph to get class name for the layer.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        list[str]: A list of shift negative activation layers.
    
    """
    negative_activation_functions = [PReLU, prelu, 
                                     ELU, elu, 
                                     Hardswish, hardswish, 
                                     SiLU, silu, 
                                     GELU, gelu]
    
    negative_activations = []

    for n in float_graph.nodes:
        if(n.layer_class in negative_activation_functions):
            # Print to Console
            if(len(negative_activations) == 0):
                Logger.warning("There are activations that contain negative values. Refer to the troubleshooting manual of \"Shift Negative Activation\".")
            Logger.warning("{}={}".format(n.name, n.layer_class.__name__))

            negative_activations.append(n.name)

    return negative_activations

def _compute_activations(name: str, activations: dict):
    """
    Creates a hook function to capture the activations of a layer.

    Args:
        name (str): The name of the layer.
        activations (dict): The dictionary to store the activations.

    Returns:
        hook (function): The hook function to register with the layer.
    """
    def hook(model, input, output):
        activation = input[0].detach()

        if name not in activations.keys():
            activations[name] = []
        activations[name].append(activation)

    return hook

def judge_unbalanced_concatnation(degrade_layers: list[str], 
                                  float_model: torch.nn.Module,
                                  dataset: Callable,
                                  xquant_config: XQuantConfig) -> List[List[Tuple[str, str, str]]]:
    """
    Judge whether the layer combines layers with significantly different value ranges

    Args:
        degrade_layers (list[str]): A list of detected degrade layers.
        float_model (torch.nn.Module): The original floating-point Pytorch model.
        dataset (Callable): Representative dataset used for similarity metrics computation.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        List[List[Tuple[str, str, str]]]: A list containing layer name before concatnation, and scale adjustment. 
                                          If the layer does not match the condition, it returns None.
    """
    
    judge_results = []

    if(xquant_config.quantize_reported_dir is None):
        Logger.warning("XQuantConfig.quantize_reported_dir is not defined. Skip judging of \'Unbalanced \"concatenation\"\'.")
        return judge_results

    org_torch_add = torch.add
    org_torch_tensor_add = torch.Tensor.__add__

    concat_layers = {}
    concat_layers_add = {}
    activations_float = {}
    float_model_modules = dict([*float_model.named_modules()])
    for layer_name in degrade_layers:
        is_search = layer_name[-3:] == '_bn' or layer_name[-10:] == '_collapsed' or layer_name[:3] == 'bn_'
        if is_search:
            logdir = xquant_config.quantize_reported_dir
            tblog_names = ['initial_graph', 'after_graph_preparation', 'pre_statistics_collection_substitutions']
            tblog_to_nodename = {}

            for tblog_name in tblog_names:
                tfevent_paths = sorted(glob.glob(os.path.join(logdir, 'tensorboard_logs', tblog_name, 'events.out.tfevents.*')))
                if(len(tfevent_paths) == 0):
                    tfevent_paths = sorted(glob.glob(os.path.join(logdir, '**', 'tensorboard_logs', tblog_name, 'events.out.tfevents.*')))

                if(len(tfevent_paths) == 0):
                    Logger.warning("TensorBoard logs not found in XQuantConfig.quantize_reported_dir. Skip judging of \'Unbalanced \"concatenation\"\'.")
                    return judge_results
                tfevent = EventAccumulator(path=tfevent_paths[0])
                tfevent.Reload()

                node_names = []
                for log in str(tfevent.Graph()).splitlines():
                    if(log[:7] == '  name:'):
                        node_name = log.split('\"')[-2]
                        node_names.append(node_name)

                tblog_to_nodename[tblog_name] = node_names

            # layer_name = 'features_1_conv_0_0_bn'

            after_graph_preparation_log = tblog_to_nodename['after_graph_preparation']

            first_node, second_node = None, None

            if layer_name[-3:] == '_bn': # features_1_conv_0_0_bn
                
                target_name = layer_name[:-3] # features_1_conv_0_0
                for idx, node_name in enumerate(after_graph_preparation_log):
                    _name = node_name.split('/')[-1] # node_name: MobileNetV2/Conv2d_1/features_1_conv_0_0, _name: features_1_conv_0_0
                    
                    if _name == target_name:
                        _idx, _first_node = idx, node_name
                        _second_node = after_graph_preparation_log[_idx + 1]
                        continue

            elif layer_name[-10:] == '_collapsed':

                target_name = layer_name[:-10]

                sorted_node_names_by_len = sorted(after_graph_preparation_log, key=len, reverse=True)
                _first_node = None
                for node_name in sorted_node_names_by_len:
                    _name = node_name.split('/')[-1]
                    if target_name.startswith(_name):
                        _first_node = _name
                        break

                if(_first_node is not None):
                    for node_name in sorted_node_names_by_len:
                        _name = node_name.split('/')[-1]
                        target_name_exclude_first_node = target_name[len(_first_node)+1:]
                        if _name == target_name_exclude_first_node or _name+"_bn" == target_name_exclude_first_node:
                            _second_node = _name
                            break

            elif layer_name[:3] == 'bn_':

                target_name = layer_name[3:]
                for idx, node_name in enumerate(after_graph_preparation_log):
                    _name = node_name.split('/')[-1]
                    
                    if _name == target_name:
                        _idx, _second_node = idx, node_name
                        _first_node = after_graph_preparation_log[_idx - 1]
                        continue
            
            if _first_node is not None and _second_node is not None:
                first_node = _first_node.split('/')[-1].replace("_", ".") # features.1.conv.0.0
                second_node = _second_node.split('/')[-1].replace("_", ".") # features.1.conv.0.1
                if first_node in float_model_modules.keys() and second_node in float_model_modules.keys():
                    float_model_modules[first_node].register_forward_hook(_compute_activations(first_node, activations_float))
                    float_model_modules[second_node].register_forward_hook(_compute_activations(second_node, activations_float))

                    concat_layers[layer_name] = (first_node, second_node)
                elif first_node in float_model_modules.keys() and second_node == "add":
                    float_model_modules[first_node].register_forward_hook(_compute_activations(first_node, activations_float))
                    concat_layers_add[layer_name] = first_node

    # Hooks cannot be applied to add operations. Define temporarily wrapper functions of "torch.add" and "+" to capture values after the first_node.
    add_activations = {}
    for first_node in concat_layers_add.values():
        add_activations[first_node] = []

    def hook_add(x, y, *args, **kwargs):
        """
        Hook function to detect calls to torch.add during model execution.

        Args:
            x (torch.Tensor): The first operand in the addition operation.
            y (torch.Tensor): The second operand in the addition operation.
            *args: Additional positional arguments passed to torch.add.
            **kwargs: Additional keyword arguments passed to torch.add.

        Returns:
            torch.Tensor: The result of the addition operation.
        """
        add_result = org_torch_add(x, y, *args, **kwargs)
        for first_node in add_activations.keys():
            conv_output = activations_float.get(first_node)[-1]
            if conv_output is not None and (torch.equal(x, conv_output) or torch.equal(y, conv_output)):
                add_activations[first_node].append(add_result.detach())
                return add_result
        return add_result

    def hook_tensor_add(self, other):
        """
        Hook function to detect calls to [torch.Tensor + torch.Tensor] during model execution.

        Args:
            self (torch.Tensor): The left operand of the addition.
            other (torch.Tensor): The right operand of the addition.

        Returns:
            torch.Tensor: The result of the addition operation.
        """
        add_result = org_torch_tensor_add(self, other)
        for first_node in add_activations.keys():
            conv_output = activations_float.get(first_node)[-1]
            if conv_output is not None and (torch.equal(self, conv_output) or torch.equal(other, conv_output)):
                add_activations[first_node].append(add_result.detach())
                return add_result
        return add_result

    # Replace temporarily wrapper add functions
    torch.add = hook_add
    torch.Tensor.__add__ = hook_tensor_add

    # Perform a forward pass with the input data and capture activations
    dataset = partial(PytorchDatasetUtils.prepare_dataset, dataset=dataset, is_validation=True)
    if(( len(concat_layers) + len(concat_layers_add)) > 0):
        for data in dataset():
            with torch.no_grad():
                _ = float_model(*data)

    # Restore the original add functions
    torch.add = org_torch_add
    torch.Tensor.__add__ = org_torch_tensor_add

    for layer_name in concat_layers.keys():
        first_node, second_node = concat_layers[layer_name]

        act_first_node = activations_float.get(first_node)
        act_second_node = activations_float.get(second_node)

        all_act_first_node, all_act_second_node = torch.cat(act_first_node), torch.cat(act_second_node)
        min_act_first_node, min_act_second_node = torch.min(all_act_first_node).item(), torch.min(all_act_second_node).item()
        max_act_first_node, max_act_second_node = torch.max(all_act_first_node).item(), torch.max(all_act_second_node).item()

        # Calculate act range
        range_first_node = max_act_first_node - min_act_first_node
        range_second_node = max_act_second_node - min_act_second_node

        # Calculate ratio
        range_ratio = range_second_node / (range_first_node + 1e-10)
        scaling_formula = "first layer * {}".format(range_ratio)

        range_ratio_over1 = range_ratio if range_ratio >= 1.0 else 1/range_ratio
        th_ratio = xquant_config.threshold_ratio_unbalanced_concatenation
        if range_ratio_over1 >= th_ratio:
            # Print to Console
            if(len(judge_results) == 0):
                Logger.warning("There are unbalanced range layers concatnated. Refer to the troubleshooting manual of \'Unbalanced \"concatenation\"\'.")
            Logger.warning("first layer:{}, second layer:{}, if you add a scaling operation, recommended scaling:{}".format(first_node, second_node, scaling_formula))

            judge_results.append((first_node, second_node, scaling_formula))

    for layer_name in concat_layers_add.keys():
        first_node = concat_layers_add[layer_name]

        act_first_node = activations_float.get(first_node)
        act_second_node = add_activations[first_node]

        all_act_first_node, all_act_second_node = torch.cat(act_first_node), torch.cat(act_second_node)
        min_act_first_node, min_act_second_node = torch.min(all_act_first_node).item(), torch.min(all_act_second_node).item()
        max_act_first_node, max_act_second_node = torch.max(all_act_first_node).item(), torch.max(all_act_second_node).item()

        # Calculate act range
        range_first_node = max_act_first_node - min_act_first_node
        range_second_node = max_act_second_node - min_act_second_node

        # Calculate ratio
        range_ratio = range_second_node / (range_first_node + 1e-10)
        scaling_formula = "first layer * {}".format(range_ratio)

        range_ratio_over1 = range_ratio if range_ratio >= 1.0 else 1/range_ratio
        th_ratio = xquant_config.threshold_ratio_unbalanced_concatenation
        if range_ratio_over1 >= th_ratio:
            # Print to Console
            if(len(judge_results) == 0):
                Logger.warning("There are unbalanced range layers concatnated. Refer to the troubleshooting manual of \'Unbalanced \"concatenation\"\'.")
            Logger.warning("first layer:{}, second layer:{}, if you add a scaling operation, recommended scaling:{}".format(first_node, second_node, scaling_formula))

            judge_results.append((first_node, second_node, scaling_formula))

    return judge_results

def judge_mixed_precision_with_model_output_loss_objective(quantized_model: torch.nn.Module,
                                                           xquant_config: XQuantConfig) -> str:
    """
    Judge whether the bitwidth of the final layer is less than threshold

    Args:
        quantized_model (torch.nn.Module): The quantized Pytorch model.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        str: The name of the final layer. If the layer does not match the condition, it returns None.
    """
    threshold_bitwidth = xquant_config.threshold_bitwidth_mixed_precision_with_model_output_loss_objective

    is_mixed_precision_with_model_output_loss_objective = False

    last_layer_name, last_layer = list(quantized_model.named_children())[-2]
    if(hasattr(last_layer, "weights_quantizers")):
        bitwidth_weights = last_layer.weights_quantizers['weight'].num_bits
        if(bitwidth_weights <= threshold_bitwidth):
            is_mixed_precision_with_model_output_loss_objective = True
    else:
        bitwidth_weights = None

    last_layer_activation_name, last_layer_activation = list(quantized_model.named_children())[-1]
    if(hasattr(last_layer_activation, "activation_holder_quantizer")):
        bitwidth_activation = last_layer_activation.activation_holder_quantizer.num_bits
        if(bitwidth_activation <= threshold_bitwidth):
            is_mixed_precision_with_model_output_loss_objective = True
    else:
        bitwidth_activation = None

    if is_mixed_precision_with_model_output_loss_objective:
        # Print to Console
        Logger.warning("the quantization bitwidth of the last layer is an extremely small number. Refer to the troubleshooting manual of \'Mixed Precision with model output loss objective\'.")
        Logger.warning("bidwidth of {}:{}(Weight), {}(Activation)".format(last_layer_name, bitwidth_weights, bitwidth_activation))

        return [last_layer_name]

    return []