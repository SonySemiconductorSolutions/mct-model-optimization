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

from typing import Dict
import tensorflow as tf
import torch
import numpy as np


def weight_translation(keras_name: str, 
                       pytorch_weights_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert a keras weight name format to torch naming format, so the value of the weight can be
    retrieved from the Torch model state_dict.

    For example:
    * Keras name: model_name/layer_name/kernel:0
    is translated to:
    * Torch name: model_name.layer_name.weight

    Args:
        keras_name (str): keras weight name
        pytorch_weights_dict (Dict[str, np.ndarray]): the Torch model state_dict, as {name_str: weight value as numpy array}

    Returns:
        np.ndarray: the weight value as a numpy array

    """
    keras_name = keras_name.replace('/', '.')
    # Handle Convolution layers
    if '.depthwise_kernel:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".depthwise_kernel:0", ".weight")).transpose((2, 3, 0, 1))
    elif '.kernel:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".kernel:0", ".weight"))
        value = value.transpose((2, 3, 1, 0))
    elif '.bias:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".bias:0", ".bias"))

    # Handle normalization layers
    elif '.beta:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".beta:0", ".bias"))
    elif '.gamma:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".gamma:0", ".weight"))
    elif '.moving_mean:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".moving_mean:0", ".running_mean"))
    elif '.moving_variance:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".moving_variance:0", ".running_var"))
    else:
        value = pytorch_weights_dict.pop(keras_name)
    return value


def load_state_dict(model: tf.keras.Model, state_dict_url: str = None,
                    state_dict_torch: Dict = None):
    """
    Assign a Keras model weights according to a state_dict from the equivalent Torch model.

    Args:
        model (tf.keras.Model): A Keras model
        state_dict_url (str): the Torch model state_dict location
        state_dict_torch(Dict[str, np.ndarray]): Torch model state_dict. If not None, will be used instead of state_dict_url

    Returns:
        tf.keras.Model: Input Keras model after assigning the weights.

    """
    if state_dict_torch is None:
        assert state_dict_url is not None, "either 'state_dict_url' or 'state_dict_torch' should not be None"
        state_dict_torch = torch.hub.load_state_dict_from_url(state_dict_url, progress=False,
                                                              map_location='cpu')
    state_dict = {k: v.numpy() for k, v in state_dict_torch.items()}

    model_copy = tf.keras.models.clone_model(model)
    for layer in model_copy.layers:
        for w in layer.weights:
            w.assign(weight_translation(w.name, state_dict))

    # look for variables not assigned in torch state dict
    for k in state_dict:
        if 'num_batches_tracked' in k:
            continue
        print(f'  WARNING: {k} not assigned to keras model !!!')
    
    return model_copy
