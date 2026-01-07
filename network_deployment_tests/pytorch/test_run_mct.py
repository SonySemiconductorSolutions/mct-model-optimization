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
import os
import numpy as np
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL


class MCTTest:

    def __init__(self, 
                 tpc_version, 
                 device_type=IMX500_TP_MODEL, 
                 save_folder='./',
                 input_shape=(3, 224, 224), 
                 batch_size=1, 
                 num_calibration_iter=1, 
                 num_of_inputs=1,
                 onnx_opset_version=20):
        self.tpc_version = tpc_version
        self.device_type = device_type
        self.save_folder = save_folder
        self.input_shape = (batch_size,) + input_shape
        self.num_calibration_iter = num_calibration_iter
        self.num_of_inputs = num_of_inputs
        self.onnx_opset_version = onnx_opset_version

    def get_input_shapes(self):
        return [self.input_shape for _ in range(self.num_of_inputs)]

    def generate_inputs(self):
        return [np.random.randn(*in_shape) for in_shape in self.get_input_shapes()]

    def representative_data_gen(self):
        for _ in range(self.num_calibration_iter):
            yield self.generate_inputs()

    def run_mct(self, float_model):
        os.makedirs(self.save_folder, exist_ok=True)
        onnx_path = os.path.join(self.save_folder, 'qmodel.onnx')

        tpc = get_target_platform_capabilities(tpc_version=self.tpc_version, device_type=self.device_type)
        
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                                                        representative_data_gen=self.representative_data_gen,
                                                                        target_platform_capabilities=tpc)

        mct.exporter.pytorch_export_model(quantized_model, save_model_path=onnx_path,
                                          repr_dataset=self.representative_data_gen,
                                          onnx_opset_version=self.onnx_opset_version)
        

def test_run_mct():

    tpc_version = os.getenv("TPC_VERSION")
    print(f"TPC VERSION: {tpc_version}")
    onnx_opset_version = os.getenv("ONNX_OPSET_VERSION")
    print(f"ONNX OPSET VERSION: {onnx_opset_version}")
    
    float_model = mobilenet_v2()
    save_folder = './mobilenet_pt'

    MCTTest(tpc_version=tpc_version, device_type=IMX500_TP_MODEL, save_folder=save_folder, 
            onnx_opset_version=int(onnx_opset_version)).run_mct(float_model)