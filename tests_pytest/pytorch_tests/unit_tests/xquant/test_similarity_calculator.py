#  Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import pytest
import torch

from model_compression_toolkit.xquant.pytorch.similarity_calculator import SimilarityCalculator
from model_compression_toolkit.xquant.pytorch.similarity_functions import PytorchSimilarityFunctions

@pytest.mark.parametrize(("inputs", "expected"), [
    # inputs: (x, y, custom_similarity_metrics)
    # expected: similarities
    ((torch.tensor([3, 3, 3], dtype=torch.float32), # x
      torch.tensor([1, 1, 1], dtype=torch.float32), # y
      {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}, #custom_similarity_metrics
     ),
     {'mse': 4.0, 'cs': 1.0, 'sqnr': 2.25, 'mae': 2.0} # similarities
    ),
    # Expected MSE  = mean((3-1)^2, (3-1)^2, (3-1)^2) = 4.0
    # Expected CS   = dot([3, 3, 3], [1, 1, 1]) / (||[3, 3, 3]|| * ||[1, 1, 1]||) = 9.0 / 9.0 = 1.0
    # Expected SQNR = mean([3^2, 3^2, 3^2]) / mean((3-1)^2, (3-1)^2, (3-1)^2) = 9.0 / 4.0 = 2.25
    # Expected MAE  = mean(|3-1|, |3-1|, |3-1|) = 2.0
  
    (([torch.tensor([3, 3, 3], dtype=torch.float32), torch.tensor([3, 3, 3], dtype=torch.float32)], # x
      [torch.tensor([1, 1, 1], dtype=torch.float32), torch.tensor([1, 1, 1], dtype=torch.float32)], # y
      {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}, #custom_similarity_metrics
     ),
     {'mse': 0.0, 'cs': 0.0, 'sqnr': 0.0, 'mae': 0.0} # similarities
    ),
    (((torch.tensor([3, 3, 3], dtype=torch.float32), torch.tensor([3, 3, 3], dtype=torch.float32)), # x
      (torch.tensor([1, 1, 1], dtype=torch.float32), torch.tensor([1, 1, 1], dtype=torch.float32)), # y
      {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}, #custom_similarity_metrics
     ),
     {'mse': 0.0, 'cs': 0.0, 'sqnr': 0.0, 'mae': 0.0} # similarities
    ),
    (({"x1":torch.tensor([3, 3, 3], dtype=torch.float32), "x2":torch.tensor([3, 3, 3], dtype=torch.float32)}, # x
      {"y1":torch.tensor([1, 1, 1], dtype=torch.float32), "y2":torch.tensor([1, 1, 1], dtype=torch.float32)}, # y
      {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}, #custom_similarity_metrics
     ),
     {'mse': 0.0, 'cs': 0.0, 'sqnr': 0.0, 'mae': 0.0} # similarities
    ),
    ((torch.tensor([3, 3, 3], dtype=torch.float32), # x
      None, # y
      {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}, #custom_similarity_metrics
     ),
     {'mse': 0.0, 'cs': 0.0, 'sqnr': 0.0, 'mae': 0.0} # similarities
    ),
    ((None, # x
      torch.tensor([1, 1, 1], dtype=torch.float32), # y
      {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}, #custom_similarity_metrics
     ),
     {'mse': 0.0, 'cs': 0.0, 'sqnr': 0.0, 'mae': 0.0} # similarities
    ),
    # If the types of inputs are not Tensor, similarities are 0.0.
    ],
)
def test_compute_similarities(inputs, expected):
    x = inputs[0]
    y = inputs[1]
    custom_similarity_metrics = inputs[2]

    similarity_calculator = SimilarityCalculator(dataset_utils=None,
                                                 model_folding=None,
                                                 similarity_functions=PytorchSimilarityFunctions(),
                                                 model_analyzer_utils=None,
                                                 device=None)
    similarity_metrics_to_compute = similarity_calculator.similarity_functions.get_default_similarity_metrics()
    similarity_metrics_to_compute.update(custom_similarity_metrics)

    output_results = similarity_calculator.compute_tensors_similarity((x, y), similarity_metrics_to_compute)

    assert output_results == expected
