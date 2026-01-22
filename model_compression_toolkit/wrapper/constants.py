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

# TPC parameters
FW_NAME = 'fw_name'
SDSP_VERSION = 'sdsp_version'

# QuantizationConfig parameters
ACTIVATION_ERROR_METHOD = 'activation_error_method'
WEIGHTS_BIAS_CORRECTION = 'weights_bias_correction'
Z_THRESHOLD = 'z_threshold'
LINEAR_COLLAPSING = 'linear_collapsing'
RESIDUAL_COLLAPSING = 'residual_collapsing'

# MixedPrecisionQuantizationConfig parameters
DISTANCE_WEIGHTING_METHOD = 'distance_weighting_method'
NUM_OF_IMAGES = 'num_of_images'
USE_HESSIAN_BASED_SCORES = 'use_hessian_based_scores'

# ResourceUtilization parameters
WEIGHTS_COMPRESSION_RATIO = 'weights_compression_ratio'

# Resource utilization data parameters
IN_MODEL = 'in_model'
REPRESENTATIVE_DATA_GEN = 'representative_data_gen'
CORE_CONFIG = 'core_config'
TARGET_PLATFORM_CAPABILITIES = 'target_platform_capabilities'

# PTQ/GPTQ parameters
TARGET_RESOURCE_UTILIZATION = 'target_resource_utilization'
IN_MODULE = 'in_module'

# GPTQ specific parameters
GPTQ_CONFIG = 'gptq_config'
MODEL = 'model'

# GPTQ parameters
N_EPOCHS = 'n_epochs'
OPTIMIZER = 'optimizer'

# low_bit_quantizer_ptq
CONVERTER_VER = 'converter_ver'
LEARNING_RATE = 'learning_rate'

# Export parameters
SAVE_MODEL_PATH = 'save_model_path'

# default compression ratio
DEFAULT_COMPRESSION_RATIO = 0.75
