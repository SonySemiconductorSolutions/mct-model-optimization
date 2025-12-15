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
import importlib
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.constants import TPC_V1_0


def generate_qnnpack_tpc(tpc_version: str) -> TargetPlatformCapabilities:
    """
    Retrieves target platform capabilities model based on the specified tpc version.

    Args:
        tpc_version (str): The version of the TPC to use.

    Returns:
        TargetPlatformCapabilities: The TargetPlatformCapabilities object based on the specified version.
    """

    # Organize all tpc versions into tpcs_dict.
    tpcs_dict = {
        TPC_V1_0: "model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.v1_0.tpc",
    }

    msg = (f"Error: The specified tpc version '{tpc_version}' is not valid. "
         f"Available versions are: {', '.join(tpcs_dict.keys())}. "
         "Please ensure you are using a supported tpc version.")
    assert tpc_version in tpcs_dict, msg

    tpc_func = importlib.import_module(tpcs_dict[tpc_version])
    return getattr(tpc_func, "get_tpc")()