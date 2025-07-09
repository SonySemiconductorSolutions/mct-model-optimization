# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from unittest.mock import Mock

import pytest

from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.target_platform_capabilities import OpQuantizationConfig


class TestNodeActivationConfig:

    def _get_op_config(self, qe, qp):
        return Mock(spec=OpQuantizationConfig,
                    activation_quantization_method=Mock(),
                    activation_n_bits=5,
                    enable_activation_quantization=qe,
                    quantization_preserving=qp,
                    signedness=Mock())

    def test_config(self):
        with pytest.raises(ValueError,
                           match="can't have both enable_activation_quantization and quantization_preserving enabled"):
            NodeActivationQuantizationConfig(self._get_op_config(True, True))

        cfg = NodeActivationQuantizationConfig(self._get_op_config(False, False))
        assert cfg.quant_mode == ActivationQuantizationMode.NO_QUANT
        self._assert_unset_acfg(cfg)

        op_cfg = self._get_op_config(True, False)
        cfg = NodeActivationQuantizationConfig(op_cfg)
        assert cfg.quant_mode == ActivationQuantizationMode.QUANT
        assert cfg.activation_n_bits == 5
        assert cfg.activation_quantization_method == op_cfg.activation_quantization_method
        assert cfg.signedness == op_cfg.signedness

        cfg = NodeActivationQuantizationConfig(self._get_op_config(False, True))
        assert cfg.quant_mode == ActivationQuantizationMode.PRESERVE_QUANT
        self._assert_unset_acfg(cfg)

    @pytest.mark.parametrize('mode', [ActivationQuantizationMode.NO_QUANT,
                                      ActivationQuantizationMode.PRESERVE_QUANT,
                                      ActivationQuantizationMode.FLN_NO_QUANT])
    def test_set_quant_mode(self, mode):
        cfg = NodeActivationQuantizationConfig(self._get_op_config(True, False))
        cfg.set_quant_mode(mode)
        assert cfg.quant_mode == mode
        # lose irrelevant config
        self._assert_unset_acfg(cfg)

        # after losing the config cannot set quant back
        for qmode in [ActivationQuantizationMode.QUANT, ActivationQuantizationMode.FLN_QUANT]:
            with pytest.raises(ValueError, match=f'Cannot change quant_mode to {qmode.name} from {mode.name}'):
                cfg.set_quant_mode(qmode)

        with pytest.raises(RuntimeError, match='quant_mode cannot be set directly'):
            cfg.quant_mode = ActivationQuantizationMode.NO_QUANT

    def test_set_quant_config_attribute(self):
        cfg = NodeActivationQuantizationConfig(self._get_op_config(True, False))

        assert cfg.activation_n_bits == 5
        cfg.set_quant_config_attr('activation_n_bits', 4)
        assert cfg.activation_n_bits == 4

        with pytest.raises(AttributeError,
                           match='Parameter activation_M_bits could not be found in the node quantization config.'):
            cfg.set_quant_config_attr('activation_M_bits', 8)

        # quant_mode has a special handling
        cfg.set_quant_config_attr('quant_mode', ActivationQuantizationMode.FLN_QUANT)
        assert cfg.quant_mode == ActivationQuantizationMode.FLN_QUANT

        cfg.set_quant_config_attr('quant_mode', ActivationQuantizationMode.PRESERVE_QUANT)
        self._assert_unset_acfg(cfg)

        cfg.set_quant_config_attr('quant_mode', ActivationQuantizationMode.NO_QUANT)
        self._assert_unset_acfg(cfg)

        with pytest.raises(ValueError, match=f'Cannot change quant_mode to QUANT from NO_QUANT.'):
            cfg.set_quant_config_attr('quant_mode', ActivationQuantizationMode.QUANT)

        with pytest.raises(ValueError, match='Cannot set attribute activation_n_bits for activation with disabled '
                                             'quantization'):
            cfg.set_quant_config_attr('activation_n_bits', 5)

    @pytest.mark.parametrize('mode', [ActivationQuantizationMode.QUANT, ActivationQuantizationMode.FLN_QUANT])
    def test_set_quantization_params(self, mode):
        cfg = NodeActivationQuantizationConfig(self._get_op_config(True, False))
        cfg.set_quant_mode(mode)

        params1 = {'foo': 5, 'bar': 10}
        cfg.set_activation_quantization_param(params1)
        assert cfg.activation_quantization_params == params1

        params2 = {'baz': 42}
        cfg.set_activation_quantization_param(params2)
        assert cfg.activation_quantization_params == params2

    def _assert_unset_acfg(self, cfg: NodeActivationQuantizationConfig):
        assert cfg.activation_n_bits == 0
        assert cfg.activation_quantization_method is None
        assert cfg.signedness is None
