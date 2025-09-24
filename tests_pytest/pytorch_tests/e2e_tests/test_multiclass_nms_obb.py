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
import onnx
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, List
import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    AttributeQuantizationConfig, Signedness
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import generate_tpc
from mct_quantizers import QuantizationMethod
from edgemdt_cl.pytorch.nms_obb import MulticlassNMSOBB, NMSOBBResults
from edgemdt_cl.pytorch import load_custom_ops


def get_representative_dataset(n_iter: int):
    def representative_dataset() -> Iterator[List]:
        for _ in range(n_iter):
            yield [torch.rand(1, 3, 64, 64)]

    return representative_dataset


def get_tpc():

    att_cfg_noquant = AttributeQuantizationConfig()
    op_cfg = OpQuantizationConfig(default_weight_attr_config=att_cfg_noquant,
                                  attr_weights_configs_mapping={},
                                  activation_quantization_method=QuantizationMethod.UNIFORM,
                                  activation_n_bits=8,
                                  supported_input_activation_n_bits=8,
                                  enable_activation_quantization=False,
                                  quantization_preserving=False,
                                  fixed_scale=None,
                                  fixed_zero_point=None,
                                  simd_size=32,
                                  signedness=Signedness.AUTO)

    tpc = generate_tpc(default_config=op_cfg, base_config=op_cfg, mixed_precision_cfg_list=[op_cfg], name="test_tpc")
    return tpc


class NMSOBBModel(nn.Module):

    def __init__(self, num_classes=2, max_detections=300, score_threshold=0.001, iou_threshold=0.7):
        
        super().__init__()
        self.max_detections = max_detections

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.bbox_reg = nn.Conv2d(16, 4 * max_detections, kernel_size=1)
        self.class_reg = nn.Conv2d(16, num_classes * max_detections, kernel_size=1)
        self.angle_reg = nn.Conv2d(16, max_detections, kernel_size=1)
        self.multiclass_nms_obb = MulticlassNMSOBB(score_threshold, iou_threshold, max_detections)

    def forward(self, x):

        batch = x.size(0)
        features = self.backbone(x)
        H_prime, W_prime = features.shape[2], features.shape[3]
        
        boxes = self.bbox_reg(features)
        boxes = boxes.view(batch, self.max_detections, 4, H_prime * W_prime).mean(dim=3)
        scores = self.class_reg(features).view(batch, self.max_detections, -1, H_prime * W_prime)
        scores = F.softmax(scores.mean(dim=3), dim=2)
        angles = self.angle_reg(features)
        angles = angles.view(batch, self.max_detections, 1, H_prime * W_prime).mean(dim=3)

        nms_res = self.multiclass_nms_obb(boxes, scores, angles)
        return nms_res
    

class TestMulticlassNMSOBB():

    def test_multiclass_nms_obb(self):

        max_detections = 300
        score_threshold = 0.001
        iou_threshold = 0.7

        model = NMSOBBModel(num_classes=2, max_detections=max_detections, score_threshold=score_threshold, iou_threshold=iou_threshold)

        tpc = get_tpc()
        q_model, _ = mct.ptq.pytorch_post_training_quantization(model, 
                                                                get_representative_dataset(n_iter=1),
                                                                target_resource_utilization=None,
                                                                core_config=mct.core.CoreConfig(),
                                                                target_platform_capabilities=tpc)
        
        _, last_layer = list(q_model.named_children())[-1]

        assert isinstance(last_layer, MulticlassNMSOBB)
        assert last_layer.score_threshold == score_threshold
        assert last_layer.iou_threshold == iou_threshold
        assert last_layer.max_detections == max_detections

        dummy_x = torch.rand(1, 3, 64, 64)
        res = q_model(dummy_x)
        assert isinstance(res, NMSOBBResults)
        assert res.boxes.shape == (1, max_detections, 4) # boxes
        assert res.scores.shape == (1, max_detections) # scores
        assert res.labels.shape == (1, max_detections) # labels
        assert res.angles.shape == (1, max_detections) # angles
        assert res.n_valid.shape == (1, 1) # n_valid

        # export onnx
        onnx_model_path = './qmodel_with_nms_obb.onnx'
        mct.exporter.pytorch_export_model(model=q_model,
                                          save_model_path=onnx_model_path,
                                          repr_dataset=get_representative_dataset(n_iter=1))
        assert os.path.exists(onnx_model_path) == True

        # load onnx
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        opset_info = list(onnx_model.opset_import)[1]
        assert opset_info.domain == 'EdgeMDT' and opset_info.version == 1

        nms_obb_node = list(onnx_model.graph.node)[-1]
        assert nms_obb_node.domain == 'EdgeMDT'
        assert nms_obb_node.op_type == 'MultiClassNMSOBB'
        assert len(nms_obb_node.input) == 3
        assert len(nms_obb_node.output) == 5

        attrs = sorted(nms_obb_node.attribute, key=lambda a: a.name)
        assert attrs[0].name == 'iou_threshold'
        np.isclose(attrs[0].f, iou_threshold)
        assert attrs[1].name == 'max_detections'
        assert attrs[1].i == max_detections
        assert attrs[2].name == 'score_threshold'
        np.isclose(attrs[2].f, score_threshold)

        # check for ort
        so = load_custom_ops()
        session = ort.InferenceSession(onnx_model_path, sess_options=so)
        ort_res = session.run(output_names=None, input_feed={'input': dummy_x.numpy()})

        assert ort_res[0].shape == (1, max_detections, 4) # boxes
        assert ort_res[1].shape == (1, max_detections) # scores
        assert ort_res[2].shape == (1, max_detections) # labels
        assert ort_res[3].shape == (1, max_detections) # angles
        assert ort_res[4].shape == (1, 1) # n_valid

        for i in range(len(res)):
            assert np.allclose(res[i].detach().numpy(), ort_res[i])

        # delete onnx model
        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)
