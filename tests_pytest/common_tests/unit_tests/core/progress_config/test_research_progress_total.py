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

from model_compression_toolkit.core.common.progress_config.progress_info_controller import \
    research_progress_total


MOCK_OBJ = Mock()


def mock_core_config(
    mixed_precision_config=None
):
    core_config = Mock()
    core_config.mixed_precision_config = mixed_precision_config
    core_config.is_mixed_precision_enabled = bool(mixed_precision_config)
    
    return core_config


def mock_mixed_precision_config(
    use_hessian_based_scores=False
):
    if use_hessian_based_scores is None:
        mixed_precision_config = None
    else:
        mixed_precision_config = Mock()
        mixed_precision_config.use_hessian_based_scores = use_hessian_based_scores

    return mixed_precision_config


def mock_gptq_config(
    hessian_weights_config=None
):
    gptq_config = Mock()
    gptq_config.hessian_weights_config = hessian_weights_config

    return gptq_config


def mock_resource_utilization(
    is_any_restricted=False
):
    if is_any_restricted is None:
        resource_utilization = None
    else:
        resource_utilization = Mock()
        resource_utilization.is_any_restricted.return_value = is_any_restricted

    return resource_utilization


class TestResearchProgressTotal:

    ### PTQ (Single Precision)
    @pytest.mark.parametrize(
        "is_any_restricted, expected",
        [
            pytest.param(None,  4, id="no_ru_flag_ptq_sp_base"),
            pytest.param(False, 4, id="disable_ru_flag_ptq_sp_base"),
        ],
    )
    def test_ptq_sp(self, is_any_restricted, expected):
        core_config = mock_core_config()
        target_resource_utilization=mock_resource_utilization(is_any_restricted)

        result = research_progress_total(
            core_config=core_config,
            target_resource_utilization=target_resource_utilization
        )
        assert result == expected

    ### PTQ (Mixed Precision)
    @pytest.mark.parametrize(
        "mp_hessian_enabled, expected",
        [
            pytest.param(None,  5, id="unset_mp_cfg_ptq_mp"),
            pytest.param(False, 5, id="mp_hessian_disable_ptq_mp"),
            pytest.param(True,  6, id="mp_hessian_enable_ptq_mp"),
        ],
    )
    def test_ptq_mp(self, mp_hessian_enabled, expected):
        core_config = mock_core_config(mixed_precision_config=mock_mixed_precision_config(mp_hessian_enabled))
        result = research_progress_total(
            core_config=core_config,
            target_resource_utilization=mock_resource_utilization(True),
        )
        assert result == expected

    ### GPTQ (Single Precision)
    @pytest.mark.parametrize(
        "is_any_restricted, gptq_hessian_weights_config, expected",
        [
            pytest.param(False, None,     5, id="disable_ru_flag_gptq_sp_enable_hessian"),
            pytest.param(False, MOCK_OBJ, 6, id="disable_ru_flag_gptq_sp_disable_hessian"),
            pytest.param(None,  None,     5, id="no_ru_flag_gptq_sp_enable_hessian"),
            pytest.param(None,  MOCK_OBJ, 6, id="no_ru_flag_gptq_sp_disable_hessian"),
        ],
    )
    def test_gptq_sp(self, is_any_restricted, gptq_hessian_weights_config, expected):
        core_config = mock_core_config()
        gptq_config = mock_gptq_config(gptq_hessian_weights_config)
        target_resource_utilization=mock_resource_utilization(is_any_restricted)

        result = research_progress_total(core_config=core_config, 
                                         gptq_config=gptq_config,
                                         target_resource_utilization=target_resource_utilization)
        assert result == expected

    ### GPTQ (Mixed Precision)
    @pytest.mark.parametrize(
        "mp_hessian_enabled, gptq_hessian_weights_config, expected",
        [
            pytest.param(None,  None,     6, id="unset_mp_cfg_and_hessian_w_cfg_gptq_mp"),
            pytest.param(False, None,     6, id="all_disabled_hessian_gptq_mp"),
            pytest.param(True,  None,     7, id="enabled_mp_hessian_disabled_gptq_hessian"),
            pytest.param(None,  MOCK_OBJ, 7, id="unset_mp_cfg_and_set_hessian_w_cfg_gptq_mp"),
            pytest.param(False, MOCK_OBJ, 7, id="disabled_mp_hessian_enabled_gptq_hessian"),
            pytest.param(True,  MOCK_OBJ, 8, id="all_enabled_hessian_gptq_mp"),
        ],
    )
    def test_gptq_mp(self, mp_hessian_enabled, gptq_hessian_weights_config, expected):
        core_config = mock_core_config(mixed_precision_config=mock_mixed_precision_config(mp_hessian_enabled))
        target_resource_utilization = mock_resource_utilization(True)
        gptq_config = mock_gptq_config(gptq_hessian_weights_config)

        result = research_progress_total(
            core_config=core_config,
            target_resource_utilization=target_resource_utilization,
            gptq_config=gptq_config,
        )
        assert result == expected
