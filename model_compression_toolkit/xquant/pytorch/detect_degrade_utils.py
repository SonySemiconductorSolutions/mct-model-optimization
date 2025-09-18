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

from model_compression_toolkit.xquant import XQuantConfig
import matplotlib.pyplot as plt

def make_similarity_graph(metrics_name: str,
                          dataset_name: str,
                          intermediate_similarity: dict,
                          degrade_layers: list[str],
                          xquant_config: XQuantConfig) -> None:
    """
    Detect degrade layers by caliculated similarities by XQuant.
    And Draw and save similarity graphs.

    Args:
        metrics_name (str): Metrics name of caluclualting quantized error. (i.g. 'mse') 
        dataset_name (str): Dataset name ('repr'(Represantation dataset) or 'val'(Validation dataset))
        intermediate_similarity (dict): Quant error reports per layers.
        degrade_layers (list[str]): A list of detected degrade layers.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        None
    """

    plot_title = "quant_loss_{}_{}".format(metrics_name, dataset_name)

    # Get x,y from intermediate_similarity
    data_x = []
    data_y = []
    for key_layername in intermediate_similarity.keys():
        value_metrics = intermediate_similarity[key_layername][metrics_name]
        data_x.append(key_layername)
        data_y.append(value_metrics)
    degrade_layer_indexes = [data_x.index(layer) for layer in degrade_layers]

    # Make Graph(adjust size to num layers)
    plt.figure(figsize=(max(int(len(data_x)/3), 1), 5))
    # Draw plot with red circle markers of degrade layers
    plt.plot(data_x, data_y, markevery=degrade_layer_indexes, marker='o', markersize=20, markeredgecolor='r', markerfacecolor=[0.0, 0.0, 0.0, 0.0], markeredgewidth=3)
    plt.grid()
    # Add labels
    plt.title(plot_title)
    plt.xlabel("layer name")
    plt.ylabel(metrics_name)
    plt.xticks(rotation=90)
    # Label colors of degraded layers are red
    _, plt_x_labels = plt.xticks()
    for degrade_layer_index in degrade_layer_indexes:
        plt_x_labels[degrade_layer_index].set_color("red")
    # Add threshold line
    threshold = xquant_config.threshold_quantize_error[metrics_name]
    plt.hlines(threshold, 0, len(data_x)-1, "red", linestyles='dashed')
    plt.text(0, threshold+(max(data_y)-min(data_y))*0.02, "threshold={}".format(threshold), color="red")

    plt.tight_layout()

    # Save Image
    path_outfile = "{}/{}.png".format(xquant_config.report_dir, plot_title, dataset_name)
    plt.savefig(path_outfile)

    plt.clf()
    plt.close()

    return None