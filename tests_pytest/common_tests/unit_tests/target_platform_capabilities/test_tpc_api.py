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
from model_compression_toolkit.target_platform_capabilities.tpc_models.get_target_platform_capabilities import get_target_platform_capabilities


class APITest:
    """
    Test to verify that the API returns the correct version number.
    """

    def __init__(self, tpc_version, device_type):
        self.tpc_version = tpc_version
        self.device_type = device_type

    def get_tpc(self):
        return get_target_platform_capabilities(tpc_version=self.tpc_version,
                                                device_type=self.device_type)

    def run_test(self, expected_tpc_path, expected_tpc_version):
        tpc = self.get_tpc()
        expected_tpc_lib = importlib.import_module(expected_tpc_path)
        expected_tpc = getattr(expected_tpc_lib, "get_tpc")()
        assert tpc == expected_tpc, f"Expected tpc_version to be {expected_tpc_version}"


def test_tpc_api():
    # TPC v1.0 & imx500
    APITest(tpc_version='1.0', device_type='imx500').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1_0.tpc', expected_tpc_version='1.0')
    
    # TPC v4.0 & imx500
    APITest(tpc_version='4.0', device_type='imx500').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4_0.tpc', expected_tpc_version='4.0')

    # TPC v5.0 & imx500
    APITest(tpc_version='5.0', device_type='imx500').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v5_0.tpc', expected_tpc_version='5.0')

    # TPC v6.0 & imx500
    APITest(tpc_version='6.0', device_type='imx500').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v6_0.tpc', expected_tpc_version='6.0')

    # TPC v1.0 & tflite
    APITest(tpc_version='1.0', device_type='tflite').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1_0.tpc', expected_tpc_version='1.0')
    
    # TPC v1.0 & qnnpack
    APITest(tpc_version='1.0', device_type='qnnpack').run_test(
        expected_tpc_path='model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.v1_0.tpc', expected_tpc_version='1.0')


def test_false_tpc_api():
    # TPC v1.8
    with pytest.raises(AssertionError, match="Error: The specified tpc version '1.8' is not valid. Available "
                                             "versions are: 1.0, 4.0, 5.0, 6.0. Please ensure you are using a supported tpc version."):
        APITest(tpc_version='1.8', device_type='imx500').run_test(expected_tpc_path='', expected_tpc_version='')

    # TPC v3.0
    with pytest.raises(AssertionError, match="Error: The specified tpc version '3.0' is not valid. Available "
                                             "versions are: 1.0, 4.0, 5.0, 6.0. Please ensure you are using a supported tpc version."):
        APITest(tpc_version='3.0', device_type='imx500').run_test(expected_tpc_path='', expected_tpc_version='')

    # Device type IMX400
    with pytest.raises(AssertionError, match="Error: The specified device type 'imx400' is not valid. Available "
                                             "devices are: imx500, tflite, qnnpack. Please ensure you are using a supported device."):
        APITest(tpc_version='1.0', device_type='imx400').run_test(expected_tpc_path='', expected_tpc_version='')