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
import importlib
import pytest
from model_compression_toolkit.target_platform_capabilities.tpc_models.get_target_platform_capabilities import get_target_platform_capabilities_sdsp


class APITest:
    """
    Test to verify that the API returns the correct version number.
    """

    def __init__(self, sdsp_version):
        self.sdsp_version = sdsp_version

    def get_tpc(self):
        return get_target_platform_capabilities_sdsp(sdsp_version=self.sdsp_version)

    def run_test(self, expected_tpc_path, expected_tpc_version):
        tpc = self.get_tpc()
        expected_tpc_lib = importlib.import_module(expected_tpc_path)
        expected_tpc = getattr(expected_tpc_lib, "get_tpc")()
        assert tpc == expected_tpc, f"Expected tpc_version to be {expected_tpc_version}"


def test_sdsp_api():
    # Sdsp converter v3.14
    APITest(sdsp_version='3.14').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1_0.tpc', expected_tpc_version='1.0')
    
    # Sdsp converter v3.16
    APITest(sdsp_version='3.16').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4_0.tpc', expected_tpc_version='4.0')

    # Sdsp converter v3.17
    APITest(sdsp_version='3.17').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v5_0.tpc', expected_tpc_version='5.0')
    
    # Sdsp converter v3.18
    APITest(sdsp_version='3.18').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v5_0.tpc', expected_tpc_version='5.0')


def test_false_sdsp_api():
    # Sdsp converter v3.15
    with pytest.raises(AssertionError, match="Error: The specified sdsp converter version '3.15' is not valid. Available "
                                             "versions are: 3.14, 3.16, 3.17, 3.18. Please ensure you are using a supported sdsp converter version."):
        APITest(sdsp_version='3.15').run_test(expected_tpc_path='', expected_tpc_version='')
