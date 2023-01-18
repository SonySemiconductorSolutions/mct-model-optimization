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
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import TensorShape
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit import quantizers_infrastructure as qi, QuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.target_platform import OpQuantizationConfig, QuantizationMethod

keras = tf.keras
layers = keras.layers

DISPATCHER = "dispatcher"
CLASS_NAME = "class_name"


class IdentityQuantizer(qi.BaseKerasTrainableQuantizer):
    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        super().__init__(quantization_config,
                         qi.QuantizationTarget.Weights,
                         [qi.QuantizationMethod.POWER_OF_TWO, qi.QuantizationMethod.SYMMETRIC])

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        return

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        return inputs


class ZeroActivationsQuantizer(qi.BaseKerasTrainableQuantizer):
    def __init__(self, quantization_config: NodeActivationQuantizationConfig):
        super().__init__(quantization_config,
                         qi.QuantizationTarget.Activation,
                         [qi.QuantizationMethod.POWER_OF_TWO, qi.QuantizationMethod.SYMMETRIC])

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        return

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool = True) -> tf.Tensor:
        return inputs * 0


def dummy_fn():
    return


op_cfg = OpQuantizationConfig(QuantizationMethod.POWER_OF_TWO,
                              QuantizationMethod.POWER_OF_TWO,
                              8,
                              8,
                              True,
                              True,
                              True,
                              True,
                              1,
                              0,
                              32)
qc = QuantizationConfig()
weight_quantization_config = NodeWeightsQuantizationConfig(qc, op_cfg, dummy_fn, dummy_fn, -1)
activation_quantization_config = NodeActivationQuantizationConfig(qc, op_cfg, dummy_fn, dummy_fn)


class BaseKerasInfrastructureTest:
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):
        self.unit_test = unit_test
        self.val_batch_size = val_batch_size
        self.num_calibration_iter = num_calibration_iter
        self.num_of_inputs = num_of_inputs
        self.input_shape = (val_batch_size,) + input_shape

    def generate_inputs(self):
        return [np.random.randn(*in_shape) for in_shape in self.get_input_shapes()]

    def get_input_shapes(self):
        return [self.input_shape for _ in range(self.num_of_inputs)]

    def get_dispatcher(self, weight_quantizers=None, activation_quantizers=None):
        return qi.KerasNodeQuantizationDispatcher(weight_quantizers, activation_quantizers)

    def get_wrapper(self, layer, dispatcher):
        return qi.KerasQuantizationWrapper(layer, dispatcher)
