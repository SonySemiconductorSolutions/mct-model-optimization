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
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.tpc_models import generate_tpc_func


def get_target_platform_capabilities(tpc_version: str = '1.0',
                                     device_type: str = IMX500_TP_MODEL) -> TargetPlatformCapabilities:
    """
    Retrieves target platform capabilities model based on tpc version and the specified device type.

    Args:
        tpc_version (str): Target platform capabilities version.
        device_type (str): The type of device for the target platform.
        
    Returns:
        The TargetPlatformCapabilities object.
    """
    # Generate a function containing tpc configurations for the specified device type.
    tpc_func = generate_tpc_func(device_type=device_type)

    # Get the target platform model for the version.
    tpc_version = str(tpc_version)
    tpc = tpc_func(tpc_version=tpc_version)

    return tpc


def get_tpc_model(name: str, tpc: TargetPlatformCapabilities):
    """
    This is a utility method that just returns the TargetPlatformCapabilities that it receives, to support existing TPC API.

    Args:
        name: the name of the TargetPlatformCapabilities (not used in this function).
        tpc: a TargetPlatformCapabilities to return.

    Returns:
        The given TargetPlatformCapabilities object.

    """

    return tpc
