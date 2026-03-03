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

import model_compression_toolkit as mct

import torch
from torch import nn


class E2ETestProgressInfoCallback:
    def __init__(self):
        self.history = []
    
    def __call__(self, info):
        self.history.append(info)


def representative_data_gen():
    yield [torch.randn(1, 3, 8, 8)]


class TestPytorchProgressVisualization:

    def _build_test_model(self):

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
                self.bn = nn.BatchNorm2d(8)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        return Model()

    def _build_expected_prog_info(self, core_config, resource_utilization, gptq_config):

        expected_str_list = ["MCT Graph Preprocessing", "Statistics Collection", "Calculate Quantization Parameters"]

        if resource_utilization is not None and resource_utilization.is_any_restricted():
            if core_config.mixed_precision_config is not None and core_config.mixed_precision_config.use_hessian_based_scores:
                expected_str_list.append("Compute Hessian for Mixed Precision")
            expected_str_list.append("Research Mixed Precision")

        if gptq_config is not None:
            if gptq_config.hessian_weights_config is not None:
                expected_str_list.append("Compute Hessian for GPTQ")
            expected_str_list.append("Train with GPTQ")

        expected_str_list.append("MCT Graph Finalization")
        
        expected_components = [
            {
                "completedComponents": component,
                "totalComponents": len(expected_str_list),
                "currentComponent": idx,
            }
            for idx, component in enumerate(expected_str_list, start=1)
        ]

        return expected_components

    @pytest.mark.parametrize('is_enable_gptq_hessian', [False, True])
    @pytest.mark.parametrize('is_enable_mp_hessian', [False, True])
    @pytest.mark.parametrize('is_enable_mp', [False, True])
    @pytest.mark.parametrize('q_method', ['ptq', 'gptq'])
    def test_pytorch_progress_visualization(self, q_method, is_enable_mp, is_enable_mp_hessian, is_enable_gptq_hessian):
        if q_method == 'ptq' and is_enable_gptq_hessian:
            pytest.skip("Skipping because the combination 'ptq' x 'gptq_hessian' is invalid.")

        float_model = self._build_test_model()
        callback_func = E2ETestProgressInfoCallback()

        tpc = mct.get_target_platform_capabilities()
        core_config = mct.core.CoreConfig(debug_config=mct.core.DebugConfig(
                                                progress_info_callback=callback_func),
                                          mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(
                                                num_of_images=1,
                                                use_hessian_based_scores=is_enable_mp_hessian))
        if is_enable_mp:
            resource_utilization_data = mct.core.pytorch_resource_utilization_data(float_model, 
                                                                                   representative_data_gen, 
                                                                                   core_config=core_config, 
                                                                                   target_platform_capabilities=tpc)
            resource_utilization = mct.core.ResourceUtilization(weights_memory=resource_utilization_data.weights_memory * 0.9)
        else:
            resource_utilization = None

        if q_method == 'gptq':
            gptq_config = mct.gptq.get_pytorch_gptq_config(n_epochs=3, 
                                                           use_hessian_based_weights=is_enable_gptq_hessian,
                                                           use_hessian_sample_attention=is_enable_gptq_hessian)
        else:
            gptq_config = None


        if q_method == 'ptq':
            _, _ = mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                                              representative_data_gen=representative_data_gen,
                                                              target_resource_utilization=resource_utilization,
                                                              core_config=core_config,
                                                              target_platform_capabilities=tpc)
        elif q_method == 'gptq':
            _, _ = mct.gptq.pytorch_gradient_post_training_quantization(model=float_model,
                                                                        representative_data_gen=representative_data_gen,
                                                                        target_resource_utilization=resource_utilization,
                                                                        gptq_config=gptq_config,
                                                                        core_config=core_config,
                                                                        target_platform_capabilities=tpc)

        expected_history = self._build_expected_prog_info(core_config, resource_utilization, gptq_config)
        assert callback_func.history == expected_history
