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
from keras.applications.mobilenet_v2 import MobileNetV2

import model_compression_toolkit as mct
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL


class MCTTest:

    def __init__(self, 
                 tpc_version, 
                 device_type=IMX500_TP_MODEL, 
                 save_folder='./',
                 input_shape=(224, 224, 3), 
                 batch_size=1, 
                 num_calibration_iter=1, 
                 num_of_inputs=1):
        self.tpc_version = tpc_version
        self.device_type = device_type
        self.save_folder = save_folder
        self.input_shape = (batch_size,) + input_shape
        self.num_calibration_iter = num_calibration_iter
        self.num_of_inputs = num_of_inputs

    def get_input_shapes(self):
        return [self.input_shape for _ in range(self.num_of_inputs)]

    def generate_inputs(self):
        return [np.random.randn(*in_shape) for in_shape in self.get_input_shapes()]

    def representative_data_gen(self):
        for _ in range(self.num_calibration_iter):
            yield self.generate_inputs()

    def run_mct(self, float_model):
        os.makedirs(self.save_folder, exist_ok=True)
        keras_path = os.path.join(self.save_folder, 'qmodel.keras')

        tpc = get_target_platform_capabilities(tpc_version=self.tpc_version, device_type=self.device_type)
        
        quantized_model, _ = mct.ptq.keras_post_training_quantization(in_model=float_model,
                                                                      representative_data_gen=self.representative_data_gen,
                                                                      target_platform_capabilities=tpc)

        mct.exporter.keras_export_model(quantized_model, save_model_path=keras_path)
        

def test_run_mct():

    tpc_version = os.getenv("TPC_VERSION")
    print(f"TPC VERSION: {tpc_version}")
    
    float_model = MobileNetV2()
    save_folder = './mobilenet_tf'

    MCTTest(tpc_version=tpc_version, device_type=IMX500_TP_MODEL, save_folder=save_folder).run_mct(float_model)