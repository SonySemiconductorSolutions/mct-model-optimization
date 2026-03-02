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

from typing import Callable
from tqdm import tqdm

from model_compression_toolkit.core.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.core.common.progress_config.progress_info_controller import \
    ProgressInfoController
from model_compression_toolkit.core.common.progress_config.constants import \
    COMPLETED_COMPONENTS, TOTAL_COMPONENTS, CURRENT_COMPONENT


def check_callback_function(info):
    pass


class CheckCallBackFunction:
    def __init__(self):
        self.history = []
        self.count = 0

    def __call__(self, info):
        self.history.append({
            COMPLETED_COMPONENTS: info[COMPLETED_COMPONENTS],
            TOTAL_COMPONENTS: info[TOTAL_COMPONENTS],
            CURRENT_COMPONENT: info[CURRENT_COMPONENT],
        })
        self.count += 1


class TestProgessInfoController:

    ### Initialization Test
    @pytest.mark.parametrize(
        "total_step, callback_function, expected",
        [
            pytest.param(-1, None, None, id="unset_callback_and_no_steps"),
            pytest.param(1,  None, None, id="unset_callback_and_with_steps"),
            pytest.param(0,  CheckCallBackFunction(), None, id="set_callback_and_no_steps"),
            pytest.param(2,  CheckCallBackFunction(), ProgressInfoController, id="set_callback_and_steps"),
            pytest.param(2,  check_callback_function, ProgressInfoController, id="set_callback_function_and_steps"),
        ],
    )
    def test_progress_info_controller_initalize(self, total_step, callback_function, expected):
        controller = ProgressInfoController(
            total_step=total_step,
            progress_info_callback=callback_function,
            description='Unit Test'
        )
        
        if expected is None:
            ### Expected value verification (None)
            assert controller is expected
        else:
            ### Expected value verification (ProgressInfoController)
            assert isinstance(controller, expected)
            assert isinstance(controller.pbar, tqdm)

            ### Verify the initialization of class member variables
            assert controller.total_step == total_step
            assert controller.current_step == 0
            assert controller.description == 'Unit Test'
            assert callable(controller.progress_info_callback)

    ### Initialization Invalid Test
    @pytest.mark.parametrize(
        "callback_function",
        [
            pytest.param(30, id="set_type_is_int"),
            pytest.param('callback', id="set_type_is_str"),
            pytest.param([check_callback_function], id="set_type_is_list"),
        ],
    )
    def test_progress_info_controller_initalize_invalid(self, callback_function):
        with pytest.raises(TypeError) as err_msg:
            controller = ProgressInfoController(
                total_step=1,
                progress_info_callback=callback_function,
                description='Initialization Invalid Test'
            )
        
        ### Verify assertion error message
        assert str(err_msg.value) == \
                f"progress_info_callback must be a callable (function or callable instance)."

    ### Normal Test
    def test_progress_info_controller_update_description(self):
        controller = ProgressInfoController(
            total_step=2,
            progress_info_callback=CheckCallBackFunction(),
        )

        controller.set_description("Preprocessing")
        controller.set_description("Finalization")

        callback = controller.progress_info_callback

        ### Verify callback was called 2 times
        assert callback.count == 2

        ### Verify first call
        assert callback.history[0][COMPLETED_COMPONENTS] == "Preprocessing"
        assert callback.history[0][TOTAL_COMPONENTS] == 2
        assert callback.history[0][CURRENT_COMPONENT] == 1

        ### Verify second call
        assert callback.history[1][COMPLETED_COMPONENTS] == "Finalization"
        assert callback.history[1][TOTAL_COMPONENTS] == 2
        assert callback.history[1][CURRENT_COMPONENT] == 2

        controller.close()

        ### Verify pbar is closed
        assert controller.pbar is None
    
    ### Invalid Test
    def test_progress_info_controller_invalid_count_check(self):
        controller = ProgressInfoController(
            total_step=1,
            progress_info_callback=CheckCallBackFunction(),
            description='Invalid Test'
        )

        with pytest.raises(AssertionError) as err_msg:
            controller.set_description("Preprocessing")
            controller.set_description("Finalization")

        ### Verify assertion error message
        assert str(err_msg.value) == \
                f"current_step: 2, exceeded total_step: 1."

        ### Verify pbar is safely closed
        assert controller.pbar is None

        ### Verify callback was called 1 time
        callback = controller.progress_info_callback
        assert callback.count == 1

    ### DebugConfig Variable Test
    @pytest.mark.parametrize(
        "callback_function, expected",
        [
            pytest.param(None, None, id="unset_callback"),
            pytest.param(check_callback_function, Callable, id="set_callback_of_function"),
            pytest.param(CheckCallBackFunction(), CheckCallBackFunction, id="set_callback_of_class"),
        ],
    )
    def test_adding_debug_config_menber_variable(self, callback_function, expected):
        debug_config = DebugConfig(progress_info_callback=callback_function)

        if expected is None:
            assert debug_config.progress_info_callback == expected
        else:
            assert callable(debug_config.progress_info_callback)
