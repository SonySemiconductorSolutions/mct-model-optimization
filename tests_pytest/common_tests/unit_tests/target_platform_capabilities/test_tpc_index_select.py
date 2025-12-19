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
from model_compression_toolkit import get_target_platform_capabilities


def test_index_select():

    tpc = get_target_platform_capabilities(tpc_version='6.0', device_type='imx500')
    assert 'IndexSelect' in [opset.name for opset in tpc.operator_set]

    for opset in tpc.operator_set:
        if opset.name == 'IndexSelect':
            for qc in opset.qc_options.quantization_configurations:
                assert qc.default_weight_attr_config.enable_weights_quantization == False
                assert qc.enable_activation_quantization == False
                assert qc.quantization_preserving == True
                assert qc.supported_input_activation_n_bits == (8, 16)