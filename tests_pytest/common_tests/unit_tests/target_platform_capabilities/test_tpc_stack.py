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
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1.tpc import get_tpc as get_tpc_imx500_v1


def test_tpc_stack():

    tpc = get_tpc_imx500_v1() # only imx500 supported
    assert 'Stack' in [opset.name for opset in tpc.operator_set]

    for opset in tpc.operator_set:
        if opset.name == 'Stack':
            for qc in opset.qc_options.quantization_configurations:
                assert qc.default_weight_attr_config.enable_weights_quantization == False
                assert qc.attr_weights_configs_mapping == {}
                assert qc.enable_activation_quantization == True
                assert qc.activation_n_bits == 8
