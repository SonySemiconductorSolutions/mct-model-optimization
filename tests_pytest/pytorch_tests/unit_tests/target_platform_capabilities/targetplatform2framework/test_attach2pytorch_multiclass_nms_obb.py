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

from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import AttachTpcToPytorch
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    AttributeQuantizationConfig, Signedness
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import generate_tpc
from mct_quantizers import QuantizationMethod
from edgemdt_cl.pytorch.nms_obb import MulticlassNMSOBB


def get_tpc():
    """
    Create a target platform capabilities (TPC) configuration with no weight and activation quantization.

    Returns a TPC object for quantization tests.
    """
    att_cfg_noquant = AttributeQuantizationConfig()

    op_cfg = OpQuantizationConfig(default_weight_attr_config=att_cfg_noquant,
                                  attr_weights_configs_mapping={},
                                  activation_quantization_method=QuantizationMethod.UNIFORM,
                                  activation_n_bits=8,
                                  supported_input_activation_n_bits=2,
                                  enable_activation_quantization=False,
                                  quantization_preserving=False,
                                  fixed_scale=None,
                                  fixed_zero_point=None,
                                  simd_size=32,
                                  signedness=Signedness.AUTO)

    tpc = generate_tpc(default_config=op_cfg, base_config=op_cfg, mixed_precision_cfg_list=[op_cfg], name="test_tpc")

    return tpc


def test_attach2pytorch_nms_obb_tpc():

    tpc = get_tpc()
    tpc = load_target_platform_capabilities(tpc)

    attach2pytorch = AttachTpcToPytorch()
    fqc = attach2pytorch.attach(tpc)

    assert MulticlassNMSOBB in attach2pytorch._opset2layer['CombinedNonMaxSuppression']

    qc = fqc.layer2qco[MulticlassNMSOBB].quantization_configurations[0]

    assert qc.default_weight_attr_config.enable_weights_quantization == False
    assert qc.default_weight_attr_config.weights_n_bits == 32
    assert qc.attr_weights_configs_mapping == {}
    assert qc.enable_activation_quantization == False
    assert qc.activation_n_bits == 8
