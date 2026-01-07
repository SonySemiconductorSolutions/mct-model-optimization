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


class ConverterTest:

    def __init__(self, save_folder='./'):
        self.save_folder = save_folder

    def check_libs(self):
        # Check if Java is installed
        result = subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise SystemExit("Stopping execution: Java is not installed.")

        # Check if IMX500 Converter is installed
        result = subprocess.run(["imxconv-tf", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise SystemExit("Stopping execution: IMX500 Converter is not installed.")

    def run_converter(self):
        assert os.path.exists(os.path.join(self.save_folder, 'qmodel.keras')), f'Keras model not found in {self.save_folder}.'
        keras_path = os.path.join(self.save_folder, 'qmodel.keras')

        # Check if Java and IMX500 Converter is installed
        self.check_libs()

        # Run IMX500 Converter
        cmd = ["imxconv-tf", "-i", keras_path, "-o", self.save_folder, "--overwrite-output"]

        env_bin_path = os.path.dirname(sys.executable)
        os.environ["PATH"] = f"{env_bin_path}:{os.environ['PATH']}"
        env = os.environ.copy()

        subprocess.run(cmd, env=env, check=True)

        os.path.exists(self.save_folder + '/qmodel.pbtxt')

        # Remove the folder for the next test
        shutil.rmtree(self.save_folder)


def test_run_converter():

    save_folder = './mobilenet_tf'

    ConverterTest(save_folder=save_folder).run_converter()