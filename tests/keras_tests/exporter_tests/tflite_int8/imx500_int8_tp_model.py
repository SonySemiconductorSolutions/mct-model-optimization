# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import List, Tuple

import tensorflow as tf
from packaging import version

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, KERAS_KERNEL, BIAS_ATTR, BIAS, \
    KERAS_DEPTHWISE_KERNEL, WEIGHTS_N_BITS
from tests.common_tests.helpers.generate_test_tp_model import generate_test_op_qc, generate_test_attr_configs

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose

import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel, OpQuantizationConfig

tp = mct.target_platform


def get_tp_model(edit_weights_params_dict, edit_act_params_dict) -> TargetPlatformModel:
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()

    updated_config = base_config.clone_and_edit(attr_to_edit={KERNEL_ATTR: edit_weights_params_dict},
                                                **edit_act_params_dict)
    op_cfg_list = [updated_config]

    return generate_tp_model(default_config=updated_config,
                             base_config=updated_config,
                             mixed_precision_cfg_list=op_cfg_list,
                             name='int8_tp_model')


def get_op_quantization_configs() -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig], OpQuantizationConfig]:
    eight_bits = generate_test_op_qc(**generate_test_attr_configs())
    four_bits = eight_bits.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}},
                                          simd_size=eight_bits.simd_size * 2)
    two_bits = eight_bits.clone_and_edit({KERNEL_ATTR: {WEIGHTS_N_BITS: 2}},
                                         simd_size=eight_bits.simd_size * 4)
    mixed_precision_cfg_list = [eight_bits, four_bits, two_bits]
    default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
    return eight_bits, mixed_precision_cfg_list, default_config


def generate_tp_model(default_config: OpQuantizationConfig,
                      base_config: OpQuantizationConfig,
                      mixed_precision_cfg_list: List[OpQuantizationConfig],
                      name: str) -> TargetPlatformModel:
    default_configuration_options = schema.QuantizationConfigOptions(
        [default_config])
    generated_tpc = schema.TargetPlatformModel(
        default_configuration_options,
        tpc_minor_version=None,
        tpc_patch_version=None,
        tpc_platform_type=None,
        add_metadata=False, name=name)
    with generated_tpc:
        schema.OperatorsSet("NoQuantization",
                            tp.get_default_quantization_config_options()
                            .clone_and_edit(enable_activation_quantization=False)
                            .clone_and_edit_weight_attribute(enable_weights_quantization=False))

        mixed_precision_configuration_options = schema.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                                 base_config=base_config)

        conv = schema.OperatorsSet("Conv", mixed_precision_configuration_options)
        fc = schema.OperatorsSet("FullyConnected", mixed_precision_configuration_options)

        any_relu = schema.OperatorsSet("AnyReLU")
        add = schema.OperatorsSet("Add")
        sub = schema.OperatorsSet("Sub")
        mul = schema.OperatorsSet("Mul")
        div = schema.OperatorsSet("Div")
        prelu = schema.OperatorsSet("PReLU")
        swish = schema.OperatorsSet("Swish")
        sigmoid = schema.OperatorsSet("Sigmoid")
        tanh = schema.OperatorsSet("Tanh")
        activations_after_conv_to_fuse = schema.OperatorSetConcat([any_relu, swish, prelu, sigmoid, tanh])
        activations_after_fc_to_fuse = schema.OperatorSetConcat([any_relu, swish, sigmoid])
        any_binary = schema.OperatorSetConcat([add, sub, mul, div])
        schema.Fusing([conv, activations_after_conv_to_fuse])
        schema.Fusing([fc, activations_after_fc_to_fuse])
        schema.Fusing([any_binary, any_relu])

    return generated_tpc


def get_int8_tpc(edit_weights_params_dict={}, edit_act_params_dict={}) -> tp.TargetPlatformCapabilities:
    default_tp_model = get_tp_model(edit_weights_params_dict, edit_act_params_dict)
    return generate_keras_tpc(name='int8_tpc', tp_model=default_tp_model)


def generate_keras_tpc(name: str, tp_model: schema.TargetPlatformModel):
    keras_tpc = tp.TargetPlatformCapabilities(tp_model)

    with keras_tpc:
        tp.OperationsSetToLayers("NoQuantization", [Reshape,
                                                    tf.reshape,
                                                    Permute,
                                                    tf.transpose,
                                                    Flatten,
                                                    Cropping2D,
                                                    ZeroPadding2D,
                                                    Dropout,
                                                    MaxPooling2D,
                                                    tf.split,
                                                    tf.quantization.fake_quant_with_min_max_vars,
                                                    tf.math.argmax,
                                                    tf.shape,
                                                    tf.math.equal,
                                                    tf.gather,
                                                    tf.cast,
                                                    tf.compat.v1.gather,
                                                    tf.nn.top_k,
                                                    tf.__operators__.getitem,
                                                    tf.compat.v1.shape])
        tp.OperationsSetToLayers("Conv",
                                 [Conv2D,
                                  DepthwiseConv2D,
                                  Conv2DTranspose,
                                  tf.nn.conv2d,
                                  tf.nn.depthwise_conv2d,
                                  tf.nn.conv2d_transpose],
                                 attr_mapping={
                                     KERNEL_ATTR: DefaultDict({
                                         DepthwiseConv2D: KERAS_DEPTHWISE_KERNEL,
                                         tf.nn.depthwise_conv2d: KERAS_DEPTHWISE_KERNEL}, default_value=KERAS_KERNEL),
                                     BIAS_ATTR: DefaultDict(default_value=BIAS)})
        tp.OperationsSetToLayers("FullyConnected", [Dense],
                                 attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                               BIAS_ATTR: DefaultDict(default_value=BIAS)})
        tp.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                             tf.nn.relu6,
                                             tf.nn.leaky_relu,
                                             ReLU,
                                             LeakyReLU,
                                             tp.LayerFilterParams(Activation, activation="relu"),
                                             tp.LayerFilterParams(Activation, activation="leaky_relu")])
        tp.OperationsSetToLayers("Add", [tf.add, Add])
        tp.OperationsSetToLayers("Sub", [tf.subtract, Subtract])
        tp.OperationsSetToLayers("Mul", [tf.math.multiply, Multiply])
        tp.OperationsSetToLayers("Div", [tf.math.divide])
        tp.OperationsSetToLayers("PReLU", [PReLU])
        tp.OperationsSetToLayers("Swish", [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])
        tp.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])
        tp.OperationsSetToLayers("Tanh", [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])
    return keras_tpc
