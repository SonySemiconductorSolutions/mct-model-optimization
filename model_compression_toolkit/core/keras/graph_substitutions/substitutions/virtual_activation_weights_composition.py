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
from tensorflow.keras.layers import Dense, DepthwiseConv2D, Conv2D, Conv2DTranspose
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.substitutions.virtual_activation_weights_composition import \
    BaseVirtualActivationWeightsComposition

"""
Matches: (DepthwiseConv2D, Conv2D, Dense, Conv2DTranspose)
"""

WEIGHTS_NODE = NodeOperationMatcher(DepthwiseConv2D) | \
               NodeOperationMatcher(Conv2D) | \
               NodeOperationMatcher(Dense) | \
               NodeOperationMatcher(Conv2DTranspose)


class VirtualActivationWeightsComposition(BaseVirtualActivationWeightsComposition):
    def __init__(self):
        super().__init__(matcher_instance=WEIGHTS_NODE)
