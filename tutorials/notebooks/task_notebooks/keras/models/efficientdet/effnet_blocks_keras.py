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

# The following code was mostly duplicated from https://github.com/huggingface/pytorch-image-models
# and changed to generate an equivalent Keras model.
# Main changes:
#   * Torch layers replaced with Keras layers
#   * removed class inheritance from torch.nn.Module
#   * changed "forward" class methods with "__call__"
#   * removed processes unused in tutorial(example_effdet_keras_mixed_precision_ptq.ipynb).
# ==============================================================================

import types
from functools import partial
import tensorflow as tf

from timm.layers import DropPath, make_divisible

__all__ = [
    'DepthwiseSeparableConv', 'InvertedResidual']


def handle_name(_name):
    return '' if _name is None or _name == '' else _name


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


def get_attn(attn_type):
    if isinstance(attn_type, tf.keras.layers.Layer):
        return attn_type
    module_cls = None
    if attn_type:
        module_cls = attn_type
    return module_cls


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    s = kwargs.pop('stride', None)
    if s is not None:
        kwargs.update({'strides': s})
    d = kwargs.pop('dilation', None)
    if d is not None:
        kwargs.update({'dilation_rate': d})
    assert padding in ['valid', 'same'], 'Not Implemented'
    kwargs.setdefault('use_bias', kwargs.pop('bias', False))
    if kwargs.get('groups', -1) == in_chs:
        kwargs.pop('groups', None)
        return tf.keras.layers.DepthwiseConv2D(kernel_size, padding=padding, **kwargs)
    else:
        return tf.keras.layers.Conv2D(out_chs, kernel_size, padding=padding, **kwargs)


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding  = padding.lower()
    if pool_type == 'max':
        # return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        return tf.keras.layers.MaxPooling2D(kernel_size, strides=stride, padding=padding.lower())
    else:
        assert False, f'Unsupported pool type {pool_type}'


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    depthwise = kwargs.pop('depthwise', False)
    # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
    groups = in_channels if depthwise else kwargs.pop('groups', 1)
    m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m



class DepthwiseSeparableConv:
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, act_layer=tf.keras.layers.ReLU,
            norm_layer=tf.keras.layers.BatchNormalization, se_layer=None, drop_path_rate=0., name=None):
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type,
            groups=groups, name=name + '/conv_dw')
        self.bn1 = norm_act_layer(in_chs, name=name + '/bn1')

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer, name=name + '/se') if se_layer else None

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type, name=name + '/conv_pw')
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act, name=name + '/bn2')
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else None

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.filters)

    def __call__(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + shortcut
        return x


class InvertedResidual:
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=tf.keras.layers.ReLU,
            norm_layer=tf.keras.layers.BatchNormalization, se_layer=None, conv_kwargs=None, drop_path_rate=0.,
            name=None):
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, name=name + '/conv_pw', **conv_kwargs)
        self.bn1 = norm_act_layer(mid_chs, name=name + '/bn1')

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            groups=groups, padding=pad_type, name=name + '/conv_dw', **conv_kwargs)
        self.bn2 = norm_act_layer(mid_chs, name=name + '/bn2')

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else None

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type,
                                      name=name + '/conv_pwl', **conv_kwargs)
        self.bn3 = norm_act_layer(out_chs, apply_act=False, name=name + '/bn3')
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else None

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.filters)

    def __call__(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + shortcut
        return x


class BatchNormAct2d:
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(
            self,
            num_features,
            epsilon=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            apply_act=True,
            act_layer=tf.keras.layers.ReLU,
            act_kwargs=None,
            inplace=True,
            drop_layer=None,
            device=None,
            dtype=None,
            name=None
    ):
        assert affine, 'Not Implemented'
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)
        if act_kwargs is None:
            act_kwargs = {}
        self.act = act_layer(**act_kwargs) if apply_act else None

    def __call__(self, x):
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


def get_norm_act_layer(norm_layer, act_layer=None):
    assert isinstance(norm_layer, (type, str,  types.FunctionType, partial))
    # assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    type_name = norm_layer.__name__.lower() 
    if type_name.startswith('batchnormalization'): 
        norm_act_layer = BatchNormAct2d 

    norm_act_kwargs.setdefault('act_layer', act_layer)
    norm_act_layer = partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args
    return norm_act_layer
