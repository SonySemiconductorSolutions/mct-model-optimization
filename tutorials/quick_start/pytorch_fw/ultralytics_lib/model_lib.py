# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

"""
 Parts of this file were copied from https://github.com/ultralytics/ultralytics and modified for this project needs.

 The Licence of the ultralytics project is shown in: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
"""
import logging

import torch
from ultralytics.utils.torch_utils import initialize_weights

from tutorials.quick_start.pytorch_fw.ultralytics_lib.common_replacers import C2fModuleReplacer, \
    YOLOReplacer, TASK_MAP
from tutorials.quick_start.pytorch_fw.ultralytics_lib.detect_replacers import DetectionModelModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.common_replacers import prepare_model_for_ultralytics_val
from tutorials.quick_start.common.model_lib import BaseModelLib
from tutorials.quick_start.common.constants import MODEL_NAME, BATCH_SIZE, COCO_DATASET, VALIDATION_DATASET_FOLDER, \
    MODULE_REPLACER
from tutorials.quick_start.common.results import DatasetInfo
from tutorials.mct_model_garden.models_keras.yolov8.yolov8_preprocess import yolov8_preprocess_chw_transpose
from model_compression_toolkit.core import FolderImageLoader
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PytorchModel


class ModelLib(BaseModelLib):
    """
    A class representing ultralytics model library (https://github.com/ultralytics/ultralytics).
    """

    def __init__(self, args):
        """
         Init the ModelLib with user arguments
         Args:
             args (dict): user arguments
         """
        # Load model from ultralytics
        self.ultralytics_model = YOLOReplacer(args[MODEL_NAME])
        self.dataset_name = COCO_DATASET
        self.preprocess = yolov8_preprocess_chw_transpose
        model_weights = self.ultralytics_model.model.state_dict()

        # Replace few modules with quantization-friendly modules
        self.model = self.ultralytics_model.model
        self.model = DetectionModelModuleReplacer().replace(self.model)
        self.model = C2fModuleReplacer().replace(self.model)
        self.model = TASK_MAP[self.ultralytics_model.task][MODULE_REPLACER].replace(self.model)

        # load pre-trained weights
        initialize_weights(self.model)
        self.model.load_state_dict(model_weights)  # load weights
        super().__init__(args)

    def get_model(self):
        """
         Returns the model instance.
         """
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        """
        Create a representative dataset generator
        Args:
            representative_dataset_folder: Dataset location folder, in YOLO format (see ultralytics docs), for example: /<my_folder>/coco/images/train2017
            n_iter: number batches to run in the generator
            batch_size: number of images in each batch

        Returns:
            A generator for the representative dataset, as the MCT expects

        """
        image_data_loader = FolderImageLoader(representative_dataset_folder,
                                              preprocessing=[self.preprocess],
                                              batch_size=batch_size)
        def representative_data_gen() -> list:
            for _ in range(n_iter):
                yield [image_data_loader.sample()]

        return representative_data_gen

    def evaluate(self, model):
        """
         Evaluates the model's performance.

         Args:
             model: The model to evaluate.

         Returns:
             acc: The accuracy of the model.
             DatasetInfo: Information about the dataset used.
         """
        # Use Cuda device if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Some attributes are required for the evaluation of the quantized model
        self.ultralytics_model = prepare_model_for_ultralytics_val(self.ultralytics_model, model, quantized=isinstance(model, PytorchModel))

        # Evaluation using ultralytics interface
        if self.args[VALIDATION_DATASET_FOLDER] is not None:
            logging.warning('The provided value for "validation_dataset_folder" is ignored. '
                            'Ultralytics utilizes the dataset path specified in the coco.yaml file. '
                            'By default, the dataset path is taken from "/home/user/.config/Ultralytics/settings.yaml", depends on your operating system.')

        results = self.ultralytics_model.val(batch=int(self.args[BATCH_SIZE]))  # evaluate model performance on the validation set
        map_res = results.mean_results()[-1]
        dataset_info = DatasetInfo(self.dataset_name, 5000)
        return map_res, dataset_info



