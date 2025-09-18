#  Copyright 2025 Sony Semiconductor Solutions. All rights reserved.
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
from typing import List
import os
import tempfile

import torch

from model_compression_toolkit.xquant.pytorch.core_detect_degrade_layer import core_detect_degrade_layer
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig

SIM_MAX = 10.0
SIM_MIN = 0.0

def get_repr_similarity(mse_fix=None, cs_fix=None, sqnr_fix=None, mae_fix=None):
    
    repr_similarity = [{"mse": SIM_MIN, "cs": SIM_MAX, "sqnr": SIM_MAX, "mae":SIM_MIN}, {}]
    repr_similarity[1]["layer1"] = {"mse": 1.1, "cs": 2.1, "sqnr": 3.1, "mae":4.1}
    repr_similarity[1]["layer2"] = {"mse": 1.0, "cs": 2.0, "sqnr": 3.0, "mae":4.0}
    repr_similarity[1]["layer3"] = {"mse": 0.9, "cs": 1.9, "sqnr": 2.9, "mae":3.9}

    for metrics_name, metrics_fix in [["mse", mse_fix], ["cs", cs_fix], ["sqnr", sqnr_fix], ["mae", mae_fix]]:
        if metrics_fix is not None:
            for layer in ["layer1", "layer2", "layer3"]:
                repr_similarity[1][layer][metrics_name] = metrics_fix
    return tuple(repr_similarity)

def get_val_similarity(mse_fix=None, cs_fix=None, sqnr_fix=None, mae_fix=None):

    val_similarity = [{"mse": SIM_MIN, "cs": SIM_MAX, "sqnr": SIM_MAX, "mae":SIM_MIN}, {}]
    val_similarity[1]["layer1"] = {"mse": 1.1, "cs": 2.1, "sqnr": 3.1, "mae":4.1}
    val_similarity[1]["layer2"] = {"mse": 1.0, "cs": 2.0, "sqnr": 3.0, "mae":4.0}
    val_similarity[1]["layer3"] = {"mse": 0.9, "cs": 1.9, "sqnr": 2.9, "mae":3.9}

    for metrics_name, metrics_fix in [["mse", mse_fix], ["cs",cs_fix], ["sqnr",sqnr_fix], ["mae",mae_fix]]:
        if metrics_fix is not None:
            for layer in ["layer1", "layer2", "layer3"]:
                val_similarity[1][layer][metrics_name] = metrics_fix
    return tuple(val_similarity)

def get_xquant_config(tmpdir):
    
    xquant_config = XQuantConfig(report_dir=tmpdir,
                                threshold_quantize_error={"mse": 1.0, "cs": 2.0, "sqnr": 3.0, "mae":4.0},
                                is_detect_under_threshold_quantize_error={"mse": False, "cs": True, "sqnr": True, "mae":False}, 
                                custom_similarity_metrics={'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()})
    return xquant_config


# check for core_detect_degrade_layer
@pytest.mark.parametrize(("inputs", "expected"), [
    # inputs: (repr_similarity, val_similarity, img_filename)
    # expected: (degrade_layers)

    (({'mse_fix': None, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_mse_repr.png'), 
     ['layer1', 'layer2']), # mse_metrics_repr
    (({'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': None, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_mse_val.png'), 
     ['layer1', 'layer2']), # mse_metrics_val
    (({'mse_fix': SIM_MIN, 'cs_fix': None, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_cs_repr.png'), 
     ['layer2', 'layer3']), # cs_metrics_repr
    (({'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': None, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_cs_val.png'), 
     ['layer2', 'layer3']), # cs_metrics_val
    (({'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': None, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_sqnr_repr.png'), 
     ['layer2', 'layer3']), # sqnr_metrics_repr
    (({'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': None, 'mae_fix': SIM_MIN}, 'quant_loss_sqnr_val.png'), 
     ['layer2', 'layer3']), # sqnr_metrics_val
    (({'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': None}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_mae_repr.png'), 
     ['layer1', 'layer2']), # custom_metrics_repr
    (({'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': None}, 'quant_loss_mae_val.png'), 
     ['layer1', 'layer2']), # custom_metrics_val
])
def test_core_detect_degrade_layer(inputs, expected):

    tmpdir = tempfile.mkdtemp()
    xquant_config = get_xquant_config(tmpdir)
    xquant_config.custom_similarity_metrics = None
    
    repr_similarity = get_repr_similarity(**inputs[0])
    val_similarity = get_val_similarity(**inputs[1])
    
    result = core_detect_degrade_layer(repr_similarity,
                                       val_similarity,
                                       xquant_config)
    
    assert isinstance(result, List)

    assert result == expected
    assert os.path.exists(os.path.join(tmpdir, inputs[2])) == True


# check for core_detect_degrade_layer inverse metrics
@pytest.mark.parametrize(("inputs", "expected"), [
    # inputs: (metric_name, repr_similarity, val_similarity, img_filename)
    # expected: (degrade_layers)

    (('mse', {'mse_fix': None, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MAX, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_mse_repr.png'), 
     ['layer2', 'layer3']), # mse_metrics_repr
    (('mse', {'mse_fix': SIM_MAX, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': None, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_mse_val.png'), 
     ['layer2', 'layer3']), # mse_metrics_val
    (('cs', {'mse_fix': SIM_MIN, 'cs_fix': None, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MIN, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_cs_repr.png'), 
     ['layer1', 'layer2']), # cs_metrics_repr
    (('cs', {'mse_fix': SIM_MIN, 'cs_fix': SIM_MIN, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': None, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN}, 'quant_loss_cs_val.png'), 
     ['layer1', 'layer2']), # cs_metrics_val
    (('sqnr', {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': None, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MIN, 'mae_fix': SIM_MIN}, 'quant_loss_sqnr_repr.png'), 
     ['layer1', 'layer2']), # sqnr_metrics_repr
    (('sqnr', {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MIN, 'mae_fix': SIM_MIN}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': None, 'mae_fix': SIM_MIN}, 'quant_loss_sqnr_val.png'), 
     ['layer1', 'layer2']), # sqnr_metrics_val
    (('mae', {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': None}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MAX}, 'quant_loss_mae_repr.png'), 
     ['layer2', 'layer3']), # custom_metrics_repr
    (('mae', {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MAX}, {'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': None}, 'quant_loss_mae_val.png'), 
     ['layer2', 'layer3']), # custom_metrics_val
])
def test_core_detect_degrade_layer_inverse(inputs, expected):

    tmpdir = tempfile.mkdtemp()
    xquant_config = get_xquant_config(tmpdir)
    xquant_config.custom_similarity_metrics = None

    if inputs[0] in {'mse', 'mae'}:
        xquant_config.is_detect_under_threshold_quantize_error[inputs[0]] = True
    else:
        xquant_config.is_detect_under_threshold_quantize_error[inputs[0]] = False
    
    repr_similarity = get_repr_similarity(**inputs[1])
    val_similarity = get_val_similarity(**inputs[2])
    
    result = core_detect_degrade_layer(repr_similarity,
                                       val_similarity,
                                       xquant_config)
    
    assert isinstance(result, List)

    assert result == expected
    assert os.path.exists(os.path.join(tmpdir, inputs[3])) == True


# check for custom_metrics threshold undefined
def test_core_detect_degrade_layer_threshold_undefined():

    tmpdir = tempfile.mkdtemp()
    xquant_config = get_xquant_config(tmpdir)
    xquant_config.threshold_quantize_error = {"mse": 1.0, "cs": 2.0, "sqnr": 3.0}
    
    repr_similarity = get_repr_similarity(**{'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': None})
    val_similarity = get_val_similarity(**{'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN})
    
    result = core_detect_degrade_layer(repr_similarity,
                                       val_similarity,
                                       xquant_config)
    
    assert isinstance(result, List)

    assert result == []
    assert os.path.exists(os.path.join(tmpdir, 'quant_loss_mae_repr.png')) == False


# check for is_detect_under_threshold_quantize_error undefined
def test_core_detect_degrade_layer_is_under_detect_undefined():

    tmpdir = tempfile.mkdtemp()
    xquant_config = get_xquant_config(tmpdir)
    del xquant_config.is_detect_under_threshold_quantize_error["mae"]
    
    repr_similarity = get_repr_similarity(**{'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': None})
    val_similarity = get_val_similarity(**{'mse_fix': SIM_MIN, 'cs_fix': SIM_MAX, 'sqnr_fix': SIM_MAX, 'mae_fix': SIM_MIN})
    
    result = core_detect_degrade_layer(repr_similarity,
                                       val_similarity,
                                       xquant_config)
    
    assert isinstance(result, List)

    assert result == ['layer1', 'layer2']
    assert os.path.exists(os.path.join(tmpdir, 'quant_loss_mae_repr.png')) == True