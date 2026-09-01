# Copyright 2026 Sony Semiconductor Solutions, Inc. All rights reserved.
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
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras

from model_compression_toolkit.core import ResourceUtilization, CoreConfig
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.sensitivity_evaluation import \
    SensitivityEvaluation
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.ptq import keras_post_training_quantization
from model_compression_toolkit.target_platform_capabilities import QuantizationMethod, AttributeQuantizationConfig, \
    OpQuantizationConfig, QuantizationConfigOptions, Signedness, OperatorSetNames, \
    TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from tests_pytest._test_util.tpc_util import configure_mp_activation_opsets, configure_mp_opsets_for_kernel_bias_ops
from tests_pytest.keras_tests.keras_test_util.keras_test_mixin import KerasFwMixin


class Model(keras.Model):
    def __init__(self, input_shape):
        inputs = keras.layers.Input(input_shape[-3:])
        outputs = keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.ReLU()(outputs)
        super().__init__(inputs=inputs, outputs=outputs)


@pytest.fixture
def input_shape():
    return 1, 8, 8, 3


@pytest.fixture
def model(input_shape):
    return Model(input_shape)


def build_tpc(default_a_bit: int, conv_a_bits: list, conv_w_bits: list, fc_a_bits: list, fc_w_bits: list, bn_a_bits: list):
    default_w_cfg = AttributeQuantizationConfig(weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                                weights_n_bits=8,
                                                weights_per_channel_threshold=True,
                                                enable_weights_quantization=True)
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=default_w_cfg.clone_and_edit(enable_weights_quantization=False),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=default_a_bit,
        supported_input_activation_n_bits=[16, 8, 4, 2],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None, fixed_zero_point=None, simd_size=32, signedness=Signedness.AUTO)

    default_w_op_cfg = default_op_cfg.clone_and_edit(
        attr_weights_configs_mapping={KERNEL_ATTR: default_w_cfg, BIAS_ATTR: AttributeQuantizationConfig()}
    )
    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])

    ops1, _ = configure_mp_opsets_for_kernel_bias_ops(opset_names=[OperatorSetNames.CONV],
                                                      base_w_config=default_w_cfg, base_op_config=default_w_op_cfg,
                                                      w_nbits=conv_w_bits, a_nbits=conv_a_bits)
    ops2, _ = configure_mp_opsets_for_kernel_bias_ops(opset_names=[OperatorSetNames.FULLY_CONNECTED],
                                                      base_w_config=default_w_cfg, base_op_config=default_w_op_cfg,
                                                      w_nbits=fc_w_bits, a_nbits=fc_a_bits)
    ops3, _ = configure_mp_activation_opsets(opset_names=[OperatorSetNames.BATCH_NORM], base_op_config=default_op_cfg,
                                             a_nbits=bn_a_bits)

    tpc = TargetPlatformCapabilities(default_qco=default_cfg, tpc_platform_type='test',
                                     operator_set=ops1 + ops2 + ops3, fusing_patterns=None)
    return tpc


class TestMixedPrecisionPTQ(KerasFwMixin):
    shape = (1, 8, 8, 3)

    @pytest.fixture
    def datagen(self, input_shape):
        return self.get_basic_data_gen([input_shape])

    def test_mixed_precision_with_non_finite_sensitivity(self, model, datagen, mocker):
        """Test that Keras PTQ completes after excluding non-finite sensitivity candidates."""
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[16, 8, 4, 2],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[4, 8], bn_a_bits=[2, 4, 8, 16])
        conv_bn_node_name = f'{model.layers[1].name}_bn'
        original_compute_metric = SensitivityEvaluation.compute_metric

        def compute_metric_with_non_finite_candidate(sensitivity_evaluator, mp_a_cfg, mp_w_cfg):
            for node_name, candidate_index in {**mp_a_cfg, **mp_w_cfg}.items():
                if node_name == conv_bn_node_name and candidate_index == 0:
                    return float('nan')
            return original_compute_metric(sensitivity_evaluator, mp_a_cfg, mp_w_cfg)

        mocker.patch.object(SensitivityEvaluation,
                            'compute_metric',
                            compute_metric_with_non_finite_candidate)
        warning_mock = mocker.patch.object(Logger, 'warning')

        qmodel, user_info = self._run(model,
                                      datagen,
                                      tpc,
                                      ResourceUtilization(weights_memory=100000),
                                      eq_ru=False)

        assert user_info.mixed_precision_cfg is not None
        assert any('Ignoring mixed-precision candidates with non-finite sensitivity' in call.args[0] for call in warning_mock.call_args_list)
        assert qmodel(np.random.randn(*self.shape)) is not None

    def _run(self, model, datagen, tpc, ru, eq_ru=True, core_cfg=None):
        core_cfg = core_cfg or CoreConfig()
        qmodel, user_info = keras_post_training_quantization(model, datagen, target_resource_utilization=ru,
                                                               core_config=core_cfg, target_platform_capabilities=tpc)
        self._validate_ru(user_info, ru, eq_ru)
        return qmodel, user_info

    def _validate_ru(self, user_info, ru, equal):
        if equal:
            assert ru == user_info.final_resource_utilization
        else:
            assert ru.is_satisfied_by(user_info.final_resource_utilization)