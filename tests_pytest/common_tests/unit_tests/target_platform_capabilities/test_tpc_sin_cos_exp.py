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
import pytest
from model_compression_toolkit import get_target_platform_capabilities


@pytest.mark.parametrize("tpc_version", [
    '5.0',
    '6.0',
])
def test_sin_cos_exp(tpc_version):

    tpc = get_target_platform_capabilities(tpc_version=tpc_version)
    operators = [opset.name for opset in tpc.operator_set]
    assert 'Sin' in operators
    assert 'Cos' in operators
    assert 'Exp' in operators

    for opset in tpc.operator_set:
        if opset.name in ['Sin', 'Cos', 'Exp']:
            for qc in opset.qc_options.quantization_configurations:
                assert qc.default_weight_attr_config.enable_weights_quantization == False
                assert qc.enable_activation_quantization == True
                assert qc.supported_input_activation_n_bits == (8, 16)