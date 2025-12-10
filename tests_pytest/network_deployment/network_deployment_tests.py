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
import shutil
import subprocess
import sys
import numpy as np
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL


class NetworkDeploymentBaseTest:

    def __init__(self, 
                 tpc_version, 
                 device_type=IMX500_TP_MODEL, 
                 save_folder='./',
                 input_shape=(3, 224, 224), 
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

    def get_tpc(self):
        return get_target_platform_capabilities(tpc_version=self.tpc_version, device_type=self.device_type)

    def run_mct(self, tpc, float_model, onnx_path):
        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=float_model,
                                                                        representative_data_gen=self.representative_data_gen,
                                                                        target_platform_capabilities=tpc)

        # Save ONNX model
        mct.exporter.pytorch_export_model(quantized_model, save_model_path=onnx_path,
                                          repr_dataset=self.representative_data_gen)

    def check_libs(self):
        # Check if Java is installed
        result = subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise SystemExit("Stopping execution: Java is not installed.")

        # Check if IMX500 Converter is installed
        result = subprocess.run(["imxconv-pt", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise SystemExit("Stopping execution: IMX500 Converter is not installed.")

    def run_test(self, float_model):
        os.makedirs(self.save_folder, exist_ok=True)
        onnx_path = os.path.join(self.save_folder, 'qmodel.onnx')

        tpc = self.get_tpc()
        self.run_mct(tpc=tpc, float_model=float_model, onnx_path=onnx_path)

        # Check if Java and IMX500 Converter is installed
        self.check_libs()

        # Run IMX500 Converter
        cmd = ["imxconv-pt", "-i", onnx_path, "-o", self.save_folder, "--overwrite-output"]

        env_bin_path = os.path.dirname(sys.executable)
        os.environ["PATH"] = f"{env_bin_path}:{os.environ['PATH']}"
        env = os.environ.copy()

        subprocess.run(cmd, env=env, check=True)

        os.path.exists(self.save_folder + '/qmodel.pbtxt')

        # Remove the folder for the next test
        shutil.rmtree(self.save_folder)


def test_network_deployment_tpc_version():

    tpc_version = os.getenv("TPC_VERSION")  # Get version from GitHub CI
    float_model = mobilenet_v2()
    save_folder = './mobilenet_pt'

    NetworkDeploymentBaseTest(tpc_version=tpc_version, device_type=IMX500_TP_MODEL,
                              save_folder=save_folder).run_test(float_model)