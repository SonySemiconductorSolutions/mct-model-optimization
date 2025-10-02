# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from typing import Iterator, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras
import model_compression_toolkit as mct
from mct_quantizers import KerasActivationQuantizationHolder


def get_model():

    inputs = keras.layers.Input((32, 32, 3))
    out1 = layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    out2 = layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    outputs = tf.stack([out1, out2], -1)
    return keras.Model(inputs, outputs)


def get_representative_dataset(n_iter=1):
    
    def representative_dataset() -> Iterator[List]:
        for _ in range(n_iter):
            yield [np.random.randn(1, 32, 32, 3)]
    return representative_dataset


def test_stack():

    model = get_model()
    tpc = mct.get_target_platform_capabilities('tensorflow', 'imx500') # only imx500 supported
    q_model, _ = mct.ptq.keras_post_training_quantization(model,
                                                          get_representative_dataset(n_iter=1),
                                                          target_resource_utilization=None,
                                                          core_config=mct.core.CoreConfig(),
                                                          target_platform_capabilities=tpc)
    
    assert getattr(q_model.layers[-2], "function") is tf.stack

    stack_activation_holder = q_model.layers[-1] # activation holder for stack layer
    assert isinstance(stack_activation_holder, KerasActivationQuantizationHolder)
    assert stack_activation_holder.activation_holder_quantizer.num_bits == 8
