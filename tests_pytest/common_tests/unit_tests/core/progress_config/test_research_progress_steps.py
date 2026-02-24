#  Copyright 2026 Sony Semiconductor Solutions, Inc. All rights reserved.
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

import pytest
from unittest.mock import Mock

from dataclasses import dataclass
from model_compression_toolkit.core.common.progress_config.progress_info_controller import research_progress_total


MOCK_OBJ = object()


def mock_core_config(
    is_mixed_precision_enabled=False,
    mixed_precision_config=None
):
    @dataclass
    class MockCoreConfig:
        is_mixed_precision_enabled: bool = False
        mixed_precision_config: object = None

    return MockCoreConfig(
        is_mixed_precision_enabled=is_mixed_precision_enabled,
        mixed_precision_config=mixed_precision_config,
    )


def mock_mp_config(
    use_hessian_based_scores=None
):
    if use_hessian_based_scores is None:
        return None

    @dataclass
    class MockMixedPrecisionConfig:
        use_hessian_based_scores: bool = False

    return MockMixedPrecisionConfig(
        use_hessian_based_scores=use_hessian_based_scores)


def mock_gptq_config(
    hessian_weights_config=None
):
    if hessian_weights_config is None:
        return None

    @dataclass
    class MockGradientPTQConfig:
        hessian_weights_config: object = None

    dummy_hessian_w_config = Mock() if hessian_weights_config is MOCK_OBJ else None
    return MockGradientPTQConfig(hessian_weights_config=dummy_hessian_w_config)


def mock_resource_utilization(
    any_restricted_flag=False
):
    if any_restricted_flag is None:
        return None

    @dataclass
    class MockResourceUtilization:
        dummy_flag: bool = False

        def is_any_restricted(self):
            return self.dummy_flag

    return MockResourceUtilization(dummy_flag=any_restricted_flag)


class TestResearchProgressTotal:

    ### PTQ (Single Precision)
    @pytest.mark.parametrize(
        "expected",
        [
            pytest.param(4, id="ptq_sp_base"),
        ],
    )
    def test_ptq_sp(self, expected):
        core_config = mock_core_config()

        result = research_progress_total(core_config)
        assert result == expected

    ### PTQ (Mixed Precision)
    @pytest.mark.parametrize(
        "mp_enabled, mp_hessian_enabled, is_any_restricted, expected",
        [
            pytest.param(True,  None,  None, 5, id="no_setting_ru_config_mp_hessian_disable"),
            pytest.param(False, False, True, 5, id="mp_hessian_disable"),
            pytest.param(False, True,  True, 6, id="mp_hessian_enabled"),
            pytest.param(False, None,  True, 5, id="setting_ru_config_mp_hessian_disable"),
            pytest.param(False, True,  True, 6, id="setting_ru_config_mp_hessian_enable"),
        ],
    )
    def test_ptq_mp(self, mp_enabled, mp_hessian_enabled, is_any_restricted, expected):
        core_config = mock_core_config(
            is_mixed_precision_enabled=mp_enabled,
            mixed_precision_config=mock_mp_config(mp_hessian_enabled),
        )
        result = research_progress_total(
            core_config,
            target_resource_utilization=mock_resource_utilization(is_any_restricted),
        )
        assert result == expected

    ### GPTQ (Single Precision)
    @pytest.mark.parametrize(
        "gptq_hessian_enabled, expected",
        [
            pytest.param(False,    5, id="gptq_sp_enable_hessian"),
            pytest.param(MOCK_OBJ, 6, id="gptq_sp_disable_hessian"),
        ],
    )
    def test_gptq_sp(self, gptq_hessian_enabled, expected):
        core_config = mock_core_config()
        gptq_config = mock_gptq_config(gptq_hessian_enabled)

        result = research_progress_total(core_config=core_config, gptq_config=gptq_config)
        assert result == expected

    ### GPTQ (Mixed Precision)
    @pytest.mark.parametrize(
        "mp_enabled, mp_hessian_enabled, is_any_restricted, gptq_hessian_enabled, expected",
        [
            pytest.param(False, None, True, False,    6, id="all_disabled_hessian_gptq_mp"),
            pytest.param(True,  True, None, False,    7, id="enabled_mp_hessian_disabled_gptq_hessian"),
            pytest.param(True,  None, None, MOCK_OBJ, 7, id="disabled_mp_hessian_enabled_gptq_hessian"),
            pytest.param(False, True, True, MOCK_OBJ, 8, id="all_enabled_hessian_gptq_mp"),
        ],
    )
    def test_gptq_mp(self, mp_enabled, mp_hessian_enabled, is_any_restricted, gptq_hessian_enabled, expected):
        core_config = mock_core_config(
            is_mixed_precision_enabled=mp_enabled,
            mixed_precision_config=mock_mp_config(mp_hessian_enabled),
        )
        target_resource_utilization = mock_resource_utilization(is_any_restricted)
        gptq_config = mock_gptq_config(gptq_hessian_enabled)

        result = research_progress_total(
            core_config=core_config,
            target_resource_utilization=target_resource_utilization,
            gptq_config=gptq_config,
        )
        assert result == expected