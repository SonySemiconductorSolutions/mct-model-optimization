# Copyright 2022 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from mct_quantizers import QuantizationMethod
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc, \
    generate_mixed_precision_test_tpc, generate_tpc_with_activation_mp


def get_tpc(name, weight_bits=8, activation_bits=8,
            weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            per_channel=True):
    tpc = generate_test_tpc({'weights_n_bits': weight_bits,
                             'activation_n_bits': activation_bits,
                             'weights_quantization_method': weights_quantization_method,
                             'activation_quantization_method': activation_quantization_method,
                             'weights_per_channel_threshold': per_channel})
    return tpc


def get_16bit_tpc(name):
    tpc = generate_test_tpc({'weights_n_bits': 16,
                             'activation_n_bits': 16})
    return tpc


def get_16bit_tpc_per_tensor(name):
    tpc = generate_test_tpc({'weights_n_bits': 16,
                             'activation_n_bits': 16,
                             'weights_per_channel_threshold': False})
    return tpc


def get_quantization_disabled_keras_tpc(name):
    tp = generate_test_tpc({'enable_weights_quantization': False,
                            'enable_activation_quantization': False})
    return tp


def get_activation_quantization_disabled_keras_tpc(name):
    tp = generate_test_tpc({'enable_activation_quantization': False})
    return tp


def get_weights_quantization_disabled_keras_tpc(name):
    tp = generate_test_tpc({'enable_weights_quantization': False})
    return tp


def get_weights_only_mp_tpc_keras(base_config, default_config, mp_bitwidth_candidates_list, name):
    mp_tpc = generate_mixed_precision_test_tpc(base_cfg=base_config,
                                                    default_config=default_config,
                                                    mp_bitwidth_candidates_list=mp_bitwidth_candidates_list)
    return mp_tpc


def get_tpc_with_activation_mp_keras(base_config, default_config, mp_bitwidth_candidates_list, name, custom_opsets={}):
    mp_tpc = generate_tpc_with_activation_mp(base_cfg=base_config,
                                                  default_config=default_config,
                                                  mp_bitwidth_candidates_list=mp_bitwidth_candidates_list,
                                                  custom_opsets=list(custom_opsets.keys()))

    return mp_tpc
