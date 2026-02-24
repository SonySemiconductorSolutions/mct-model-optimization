
# Copyright 2026 Sony Semiconductor Solutions, Inc. All rights reserved.
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

from typing import Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from tqdm import tqdm

from model_compression_toolkit.core.common.progress_config.constants import \
    COMPLETED_COMPONENTS, TOTAL_COMPONENTS, CURRENT_COMPONENT

if TYPE_CHECKING:    # pragma: no cover
    from model_compression_toolkit.core import CoreConfig
    from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
    from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization


@dataclass
class ProgressInfoController:
    """
    A unified progress bar controller class.
    Support single progress bar.
    
    Attributes:
        total_step: Total number of processing steps.
        description: Description for the progress bar.
        current_step: Current step number (starts from 0, incremented by set_description()).
        callback: User-defined callback function.
    """
    total_step: int = field(default=0)
    current_step: int = field(default=0)
    description: str = field(default="Model Compression Toolkit Progress Infomation")
    progress_info_callback: Optional[Callable] = field(default=None)

    def __new__(cls, *args, **kwargs):
        """
        Create or skip instantiation based on the enable flag.
        Returns None when progress display should be disabled.
        """
        if kwargs.get('progress_info_callback') is None or \
           kwargs.get('total_step') <= 0:
            return None
        return super().__new__(cls)

    def __post_init__(self):
        """Create progress bar after initialization."""
        # Initial single bar mode (position=0, top of screen)
        self.pbar = tqdm(
            total=self.total_step,
            desc=self.description,
            position=0,
            leave=False,
            unit='step',
            dynamic_ncols=True,
            bar_format='{l_bar}{bar:}|'
        )

    def set_description(self, description: str):
        """
        Update progress bar description.
        Automatically increments step number each time set_description is called,
        displaying in "Step X/Y: ..." format.
        
        Args:
            description: New description text ("Step X/Y: " is automatically added).
        """
        self.current_step += 1
        formatted_description = f"Step {self.current_step}/{self.total_step}: {description}"
        
        try:
            assert self.current_step <= self.total_step, \
                    f"current_step: {self.current_step}, exceeded total_step: {self.total_step}."
        except AssertionError:
            self.close()
            raise

        self.pbar.set_description(formatted_description, refresh=False)
        self.pbar.update()

        progress_info = {
            COMPLETED_COMPONENTS: description,
            TOTAL_COMPONENTS: self.total_step,
            CURRENT_COMPONENT: self.current_step
        }
        self.progress_info_callback(progress_info)

    def close(self):
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def research_progress_total(
    core_config: 'CoreConfig',
    target_resource_utilization: 'ResourceUtilization' = None,
    gptq_config: 'GradientPTQConfig' = None,
) -> int:
    """
    Check whether specific processing will be executed based on input arguments
    and calculate the total number of processing steps.
    
    Processing step breakdown:
    1. Preprocessing (required)
    2. Statistics calculation (required)
    3. Weight parameter calculation (required)
    4. Hessian calculation (when GPTQ or specific settings enabled)
    5. MP calculation (when Mixed Precision enabled)
    6. Post-processing ~ conversion to exportable model (required)
    
    Args:
        core_config: CoreConfig object.
        target_resource_utilization: ResourceUtilization object (used for Mixed Precision determination).
        gptq_config: GPTQ configuration object.
        
    Returns:
        Total number of processing steps.
    """
    # Base required steps: preprocessing, statistics, weight params, post-processing
    total_steps = 4

    # Add MP calculation step (when Mixed Precision enabled)
    if target_resource_utilization is not None and \
       target_resource_utilization.is_any_restricted():
        total_steps += 1

        # Add Hessian step (when Mixed Precision with Hessian enabled)
        if core_config.mixed_precision_config is not None and \
           core_config.mixed_precision_config.use_hessian_based_scores:
            total_steps += 1

    # Add GPTQ training step (when GPTQ is enabled)
    if gptq_config is not None:
        total_steps += 1

        # Add Hessian step (when GPTQ with Hessian enabled)
        if gptq_config.hessian_weights_config is not None:
            total_steps += 1

    return total_steps
