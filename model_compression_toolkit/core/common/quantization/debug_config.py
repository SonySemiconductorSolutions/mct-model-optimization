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

from dataclasses import dataclass, field
from typing import List, Callable

from model_compression_toolkit.core.common.network_editors.edit_network import EditRule


@dataclass
class DebugConfig:
    """
    A dataclass for MCT core debug information.

    Args:
        analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is
         enabled) or not. Can be used to pinpoint problematic layers in the quantization process.
        network_editor (List[EditRule]): A list of rules and actions to edit the network for quantization.
        simulate_scheduler (bool): Simulate scheduler behavior to compute operators' order and cuts.
        bypass (bool): A flag to enable MCT bypass, which skips MCT runner and returns the input model unchanged.
        progress_info_callback (Callable): A user-defined callback function for retrieving progress information.

    About progress_info_callback

        The `progress_info_callback` parameter in `DebugConfig` enables the following features and allows users to retrieve progress information when a callback function is configured:

        - The callback function can receive MCT progress information.
        - A progress bar is displayed in the CUI, allowing users to visualize how much processing has been completed while MCT is running.

        If no callback function is set, these features are disabled and the behavior and output remain unchanged.
        Examples of how to create a callback function to enable these features are provided in the Examples section.

    Examples:

        Create a callable callback function.
        When defining the callback, make sure it accepts a dictionary representing the current processing state as an argument.

        Example 1: Use a class to keep track of the processing history.

        >>> class ProgressInfoCallback:
        ...    def __init__(self):
        ...        self.history = []
        ...
        ...    def __call__(self, info):
        ...        current = info["currentComponent"]
        ...        total = info["totalComponents"]
        ...        component_name = info["completedComponents"]
        ...
        ...        self.history.append({
        ...            "component_name": component_name,
        ...            "current": current,
        ...            "total": total
        ...        })
        ...
        >>> progress_info_callback = ProgressInfoCallback()


        Example 2: Use a function to output the progress percentage and processing name to standard error (stderr).

        >>> def progress_info_callback(info):
        ...    current = info["currentComponent"]
        ...    total = info["totalComponents"]
        ...    component_name = info["completedComponents"]
        ...
        ...    progress_percent = (current / total * 100.0)
        ...
        ...    print(f"[{current}/{total}] {progress_percent:6.2f}% {component_name}",
        ...          file=__import__('sys').stderr, flush=True)

        From the processing state dictionary, you can retrieve information using the following keys:

        .. list-table:: Keys in the processing state dictionary
            :header-rows: 1

            *   - Parameter Key
                - Value Type
                - Description
            *   - "currentComponent"
                - int
                - Current processing step
            *   - "totalComponents"
                - int
                - Total number of processing steps
            *   - "completedComponents"
                - str
                - Name of the component currently being processed

        Import MCT and configure DebugConfig with the callback function you created.
        Configure CoreConfig with this DebugConfig and use it.

        >>> import model_compression_toolkit as mct
        >>> debug_config = mct.core.DebugConfig(progress_info_callback=progress_info_callback)
        >>> core_config = mct.core.CoreConfig(debug_config=debug_config)

        .. important::
            If a callback function is configured, the GPTQ data iteration progress bar is disabled and not displayed.

    """

    analyze_similarity: bool = False
    network_editor: List[EditRule] = field(default_factory=list)
    simulate_scheduler: bool = False
    bypass: bool = False
    progress_info_callback: Callable = None
