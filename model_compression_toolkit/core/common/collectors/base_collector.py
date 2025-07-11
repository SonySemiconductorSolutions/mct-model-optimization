# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from abc import ABC, abstractmethod
import numpy as np
from model_compression_toolkit.logger import Logger


class BaseCollector(ABC):
    """
    Base class for statistics collection object.
    """

    def __init__(self):
        # When manipulation statistics in a granularity they were not collected by, the data is invalid.
        self.is_legal = True

    @abstractmethod
    def scale(self, scale_factor: np.ndarray):
        """
        Scale all statistics in collector by some factor.
        Args:
            scale_factor: Factor to scale all collector's statistics by.

        """

        raise NotImplemented(
            f'{self.__class__.__name__} needs to implement scale operation for its state.')  # pragma: no cover

    @abstractmethod
    def shift(self, shift_value: np.ndarray):
        """
        Shift all statistics in collector by some value.
        Args:
            shift_value: Value to shift all collector's statistics by.

        """

        raise NotImplemented(
            f'{self.__class__.__name__} needs to implement shift operation for its state.')  # pragma: no cover

    def update_legal_status(self, is_illegal: bool):
        """
        If statistics were manipulated in a granularity they were not collected by, the data is invalid,
        and its legal status should be tracked after each manipulation.
        Args:
            is_illegal: Whether current info is invalid or not.

        """

        self.is_legal = self.is_legal and not is_illegal

    def validate_data_correctness(self):
        """
        Verify the collector's statistics were manipulated in a granularity they were collected by.
        If the statistics are invalid, an exception is raised.
        """

        if not self.is_legal:
            Logger.critical('The data is invalid.'
                            f'{self.__class__.__name__} was collected per-tensor but received data manipulated per-channel.')  # pragma: no cover
