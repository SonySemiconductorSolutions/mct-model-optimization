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
import copy
from typing import List
from unittest.mock import Mock

import pytest

from mct_quantizers import QuantizationMethod

from model_compression_toolkit.core.common.framework_info import ChannelAxisMapping
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeWeightsQuantizationConfig, WeightsAttrQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import Signedness, OpQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.constants import POSITIONAL_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.v1 import AttributeQuantizationConfig


class TestNodeWeightsAttrConfig:
    @pytest.mark.parametrize('method, nbits, per_channel, enabled', [
        (QuantizationMethod.POWER_OF_TWO, 5, True, True),
        (QuantizationMethod.SYMMETRIC, 7, False, True),
        (QuantizationMethod.SYMMETRIC, 7, False, False),
    ])
    def test_config(self, method, nbits, per_channel, enabled):
        input_cfg = AttributeQuantizationConfig(
            weights_quantization_method=method,
            weights_n_bits=nbits,
            weights_per_channel_threshold=per_channel,
            enable_weights_quantization=enabled,
            lut_values_bitwidth=None)

        cfg = WeightsAttrQuantizationConfig(input_cfg, weights_channels_axis=ChannelAxisMapping(2, 3))
        assert cfg.enable_weights_quantization == enabled
        if enabled:
            assert cfg.weights_quantization_method == method
            assert cfg.weights_n_bits == nbits
            assert cfg.weights_per_channel_threshold == per_channel
            assert cfg.weights_channels_axis == ChannelAxisMapping(2, 3)
        else:
            assert_unset_attr_config(cfg)

        # disable quantization
        cfg.disable_quantization()
        assert cfg.enable_weights_quantization is False
        assert_unset_attr_config(cfg)

        with pytest.raises(RuntimeError, match='enable_quantization should not be set directly'):
            cfg.enable_weights_quantization = False

    def test_set_quantization_param(self):
        input_cfg = AttributeQuantizationConfig(enable_weights_quantization=True)
        cfg = WeightsAttrQuantizationConfig(input_cfg)
        params1 = {'foo': 5, 'bar': 10}
        cfg.set_weights_quantization_param(params1)
        assert cfg.weights_quantization_params == params1

        params2 = {'baz': 42}
        cfg.set_weights_quantization_param(params2)
        assert cfg.weights_quantization_params == params2

    def test_unsupported_lut(self):
        input_cfg = AttributeQuantizationConfig(enable_weights_quantization=True, lut_values_bitwidth=5)
        with pytest.raises(ValueError, match='None-default lut_values_bitwidth in AttributeQuantizationConfig '
                                             'is not supported.'):
            WeightsAttrQuantizationConfig(input_cfg)


def assert_unset_attr_config(cfg: WeightsAttrQuantizationConfig):
    assert cfg.weights_quantization_method is None
    assert cfg.weights_n_bits == 0
    assert cfg.weights_per_channel_threshold is None
    assert cfg.weights_channels_axis is None


class TestWeightsQuantizationConfig:
    def _create_weights_attr_quantization_config(self, weights_n_bits: int) -> AttributeQuantizationConfig:
        """
        Helper method to create a weights attribute quantization configuration.

        Args:
            weights_n_bits (int): Number of bits to use for quantizing weights.

        Returns:
            AttributeQuantizationConfig: Holds the quantization configuration of a weight attribute of a layer.
        """
        weights_attr_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
            weights_n_bits=weights_n_bits,
            weights_per_channel_threshold=False,
            enable_weights_quantization=True,
            lut_values_bitwidth=None)
        return weights_attr_config

    def _create_node_weights_op_cfg(
            self,
            pos_weight_attr: List[str],
            pos_weight_attr_config: List[AttributeQuantizationConfig],
            def_weight_attr_config: AttributeQuantizationConfig) -> OpQuantizationConfig:
        """
        Helper method to create a Node Weights OpQuantizationConfig with a default weights
        attribute config and a specific weight attribute.

        Args:
            pos_weight_attr (List[str]): List of names for specific weight attributes.
            pos_weight_attr_config (List[AttributeQuantizationConfig]): Corresponding list of quantization configs
                                                                    for the specific attributes.
            def_weight_attr_config (AttributeQuantizationConfig): Default quantization config for the weights.

        Returns:
            OpQuantizationConfig: Class to configure the quantization parameters of an operator.
        """
        attr_weights_configs_mapping = dict(zip(pos_weight_attr, pos_weight_attr_config))

        op_cfg = OpQuantizationConfig(
            default_weight_attr_config=def_weight_attr_config,
            attr_weights_configs_mapping=attr_weights_configs_mapping,
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=True,
            fixed_scale=None,
            fixed_zero_point=None,
            simd_size=None,
            signedness=Signedness.AUTO
        )
        return op_cfg

    def test_node_weights_quantization_config_op_cfg_mapping(self):
        """
        Test case for verifying that the positional weight attribute is correctly mapped and
        configured in the NodeWeightsQuantizationConfig.
        """
        positional_weight_attr = 0
        weights_n_bits = 8
        pos_weights_n_bits = 16

        def_weight_attr_config = self._create_weights_attr_quantization_config(weights_n_bits)
        pos_weight_attr_config = self._create_weights_attr_quantization_config(pos_weights_n_bits)

        # Ensure the configs have different weights bit widths.
        assert def_weight_attr_config.weights_n_bits != pos_weight_attr_config.weights_n_bits

        op_cfg = self._create_node_weights_op_cfg(pos_weight_attr=[POSITIONAL_ATTR],
                                                  pos_weight_attr_config=[pos_weight_attr_config],
                                                  def_weight_attr_config=def_weight_attr_config)

        # Check that positional weights attribute config differs from default config.
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            POSITIONAL_ATTR].weights_n_bits

        weights_quant_cfg = NodeWeightsQuantizationConfig(op_cfg=op_cfg,
                                                          weights_channels_axis=Mock(),
                                                          node_attrs_list=[positional_weight_attr])

        # Check if the positional weight attribute was properly assigned in the positional attributes configuration
        # mapping.
        assert weights_quant_cfg.pos_attributes_config_mapping[
                   positional_weight_attr].weights_n_bits == pos_weight_attr_config.weights_n_bits

        # Test using the positional attribute as the key rather than POS_ATTR; this mismatch should cause
        # NodeWeightsQuantizationConfig to fall back to the default weights attribute configuration instead of
        # applying the specific one.
        op_cfg = self._create_node_weights_op_cfg(pos_weight_attr=[str(positional_weight_attr)],
                                                  pos_weight_attr_config=[pos_weight_attr_config],
                                                  def_weight_attr_config=def_weight_attr_config)

        # Check that positional weights attribute config differs from default config.
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            str(positional_weight_attr)].weights_n_bits

        weights_quant_cfg = NodeWeightsQuantizationConfig(op_cfg=op_cfg,
                                                          weights_channels_axis=Mock(),
                                                          node_attrs_list=[positional_weight_attr])

        # Check if the positional weight attribute was properly assigned in the positional attributes configuration
        # mapping.
        assert weights_quant_cfg.pos_attributes_config_mapping[
                   positional_weight_attr].weights_n_bits == def_weight_attr_config.weights_n_bits

        # Add a second positional attribute with a different config.
        second_positional_weight_attr = POSITIONAL_ATTR + '_1'
        second_pos_weights_n_bits = 32
        second_pos_weight_attr_config = self._create_weights_attr_quantization_config(second_pos_weights_n_bits)

        # Confirm all three configs have different bit widths.
        assert pos_weight_attr_config.weights_n_bits != second_pos_weight_attr_config.weights_n_bits

        # Create op config with two positional attribute keys and their respective configs.
        op_cfg = self._create_node_weights_op_cfg(pos_weight_attr=[POSITIONAL_ATTR, second_positional_weight_attr],
                                                  pos_weight_attr_config=[pos_weight_attr_config,
                                                                          second_pos_weight_attr_config],
                                                  def_weight_attr_config=def_weight_attr_config)

        # Check the configs are correctly set and distinct from each other and from the default.
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            str(POSITIONAL_ATTR)].weights_n_bits
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            str(second_positional_weight_attr)].weights_n_bits
        assert op_cfg.attr_weights_configs_mapping[
                   str(POSITIONAL_ATTR)].weights_n_bits != op_cfg.attr_weights_configs_mapping[
                   str(second_positional_weight_attr)].weights_n_bits

        # Expect ValueError: multiple matching keys found for positional weights attribute.
        with pytest.raises(ValueError, match='Found multiple attribute in FQC OpConfig that are contained in the '
                                             'attribute name \'0\'.Please fix the FQC attribute names mapping such '
                                             'that each operator\'s attribute would have a unique matching name.'):
            NodeWeightsQuantizationConfig(op_cfg=op_cfg, weights_channels_axis=Mock(),
                                          node_attrs_list=[positional_weight_attr])

    def _create_wcfg(self):
        # include enabled and disabled attrs
        # include a name identical to config keys, and an extended name
        attr_weights_configs_mapping = {'foo': AttributeQuantizationConfig(enable_weights_quantization=True,
                                                                           weights_n_bits=7),
                                        'bar': AttributeQuantizationConfig(enable_weights_quantization=False)}
        default_weight_attr_config = AttributeQuantizationConfig(enable_weights_quantization=True,
                                                                 weights_n_bits=5)
        node_attrs_list = ['afooz', 'bar', 0, 1]
        wcfg = NodeWeightsQuantizationConfig(Mock(spec=OpQuantizationConfig,
                                                  attr_weights_configs_mapping=attr_weights_configs_mapping,
                                                  default_weight_attr_config=default_weight_attr_config,
                                                  simd_size=None),
                                             weights_channels_axis=ChannelAxisMapping(1, 2),
                                             node_attrs_list=node_attrs_list)
        return wcfg, node_attrs_list

    def test_has_get_set_weights_attr_config(self):
        """ Test has_attr_config, get_attr_config and set_attr_config """
        wcfg, node_attrs_list = self._create_wcfg()

        for attr in node_attrs_list:
            assert wcfg.has_attribute_config(attr) is True
        assert wcfg.has_attribute_config('baz') is False
        assert wcfg.has_attribute_config(2) is False

        assert wcfg.get_attr_config('foo').weights_n_bits == 7
        # get config should work by both long and short name
        assert wcfg.get_attr_config('afooz') == wcfg.get_attr_config('foo')
        assert wcfg.get_attr_config('bar').enable_weights_quantization is False
        assert wcfg.get_attr_config(0).weights_n_bits == 5
        assert wcfg.get_attr_config(1).weights_n_bits == 5

        new_cfg = Mock()
        wcfg.set_attr_config('afooz', new_cfg)
        assert wcfg.get_attr_config('foo') == new_cfg

        assert wcfg.get_attr_config('bar') != new_cfg
        wcfg.set_attr_config('bar', new_cfg)
        assert wcfg.get_attr_config('bar') == new_cfg

        assert wcfg.get_attr_config(1) != new_cfg
        wcfg.set_attr_config(1, new_cfg)
        assert wcfg.get_attr_config(1) == new_cfg

        # non-existing attrs
        with pytest.raises(ValueError, match='Unknown weights attr foo'):
            # set attr expects the full name
            wcfg.set_attr_config('foo', new_cfg)
        with pytest.raises(ValueError, match='Unknown weights attr 2'):
            wcfg.set_attr_config(2, new_cfg)

        # non-existing attrs with force=True
        wcfg.set_attr_config('baz', new_cfg, force=True)
        assert wcfg.get_attr_config(1) == new_cfg

        wcfg.set_attr_config(2, new_cfg, force=True)
        assert wcfg.get_attr_config(1) == new_cfg

    def test_set_quant_config_wcfg_level(self):
        """ Test set_quant_config for attributes at the weight config level. """
        wcfg, _ = self._create_wcfg()

        assert wcfg.simd_size is None
        wcfg.set_quant_config_attr('simd_size', 5)
        assert wcfg.simd_size == 5

        with pytest.raises(AttributeError):
            wcfg.set_quant_config_attr('no_such_attr', 5)

    def test_set_quant_config_attr_level(self):
        """ Test set_quant_config for attributes of weights attrs configs. """
        wcfg, _ = self._create_wcfg()

        wcfg.set_quant_config_attr('weights_n_bits', 4, attr_name='afooz')
        assert wcfg.get_attr_config('afooz').weights_n_bits == 4

        assert wcfg.get_attr_config(0).weights_n_bits == 5
        wcfg.set_quant_config_attr('weights_n_bits', 7, attr_name=1)
        assert wcfg.get_attr_config(1).weights_n_bits == 7
        # 0 is not affected
        assert wcfg.get_attr_config(0).weights_n_bits == 5

        # enable_weights_quantization has a special handling:
        foo_cfg = copy.deepcopy(wcfg.get_attr_config('afooz'))
        # True with already enabled quantization has no effect (but doesn't fail)
        wcfg.set_quant_config_attr('enable_weights_quantization', True, attr_name='afooz')
        assert wcfg.get_attr_config('afooz') == foo_cfg
        # False should reset all attrs
        wcfg.set_quant_config_attr('enable_weights_quantization', False, attr_name='afooz')
        assert_unset_attr_config(wcfg.get_attr_config('afooz'))
        # False can be set again (check that doesn't crash)
        wcfg.set_quant_config_attr('enable_weights_quantization', False, attr_name='afooz')

    def test_set_quant_config_attr_level_errors(self):
        """ Test set_quant_config for attributes of weights attrs configs. """
        wcfg, _ = self._create_wcfg()

        for attr in ['baz', 2]:
            with pytest.raises(ValueError, match=f'Weights attribute {attr} could not be found'):
                wcfg.set_quant_config_attr('weights_n_bits', 5, attr_name=attr)

        with pytest.raises(AttributeError, match='Parameter no_such_attr could not be found in the quantization config '
                                                 'of weights attribute 1'):
            wcfg.set_quant_config_attr('no_such_attr', 5, attr_name=1)

        # disabled quantization cannot be turned on (enable_weights_quantization has a special handling)
        with pytest.raises(ValueError, match=f'Cannot enable quantization for attr bar with disabled quantization.'):
            wcfg.set_quant_config_attr('enable_weights_quantization', True, attr_name='bar')
        # no other attr can be set for disabled quantization
        with pytest.raises(ValueError, match=f'Cannot set param weights_n_bits for attr bar with disabled quantization.'):
            wcfg.set_quant_config_attr('weights_n_bits', 5, attr_name='bar')

    def test_disable_all(self):
        wcfg, node_attrs_list = self._create_wcfg()
        wcfg.disable_all_weights_quantization()
        for attr in node_attrs_list:
            assert_unset_attr_config(wcfg.get_attr_config(attr))

    def test_get_all(self):
        wcfg, node_attrs_list = self._create_wcfg()
        assert sorted(wcfg.all_weight_attrs, key=lambda v: str(v)) == sorted(node_attrs_list, key=lambda v: str(v))
        cfgs = wcfg.get_all_weight_attrs_configs()
        assert cfgs == {attr: wcfg.get_attr_config(attr) for attr in node_attrs_list}
