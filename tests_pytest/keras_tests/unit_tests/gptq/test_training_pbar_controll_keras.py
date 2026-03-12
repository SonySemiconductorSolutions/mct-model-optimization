#  Copyright 2026 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.gptq.keras.gptq_training import KerasGPTQTrainer


class TestTraininPbarControllKeras:
    def _build_trainer(self, progress_info_controller):

        def _model_build_fn_return_value():
            return (Mock(layers=[]), Mock(input_scale=1))

        def _build_dummy_model():
            return Mock()

        def representative_data_gen():
            yield [np.array([1.0], dtype=np.float32)]

        fw_impl = Mock()
        fw_impl.model_builder.return_value = _model_build_fn_return_value()

        gptq_config = GradientPTQConfig(n_epochs=2,
                                        loss=Mock(),
                                        optimizer=Mock(),
                                        optimizer_rest=Mock(),
                                        train_bias=False,
                                        hessian_weights_config=None,
                                        gradual_activation_quantization_config=None,
                                        regularization_factor=0.0)

        with patch('model_compression_toolkit.gptq.common.gptq_training.get_compare_points', return_value=([], None, [], [])), \
            patch.object(KerasGPTQTrainer, 'build_gptq_model', return_value=_model_build_fn_return_value()):
            trainer = KerasGPTQTrainer(graph_float=Mock(),
                                    graph_quant=Mock(),
                                    gptq_config=gptq_config,
                                    fw_impl=fw_impl,
                                    fw_info=Mock(),
                                    representative_data_gen=representative_data_gen,
                                    progress_info_controller=progress_info_controller)

        trainer.float_model = _build_dummy_model()
        trainer.fxp_model = _build_dummy_model()

        trainer.loss_list = []
        trainer.nano_training_step = Mock(return_value=(Mock(), [[Mock()]]))
        return trainer

    def _build_tqdm_contexts(self, train_dataloader):

        def _build_pbar(callable_fn):
            pbar = MagicMock()
            pbar.__enter__.return_value = callable_fn
            pbar.__exit__.return_value = False
            return pbar

        epochs_pbar = _build_pbar(range(1))
        data_pbar = _build_pbar(train_dataloader)
        return epochs_pbar, data_pbar

    @pytest.mark.parametrize(
        'progress_info_controller, expected_disable_data_pbar',
        [
            pytest.param(Mock(), True,  id='disable_data_pbar'),
            pytest.param(None,   False, id='enable_data_pbar'),
        ],
    )
    def test_training_pbar_controll_keras(self, progress_info_controller, expected_disable_data_pbar):
        trainer = self._build_trainer(progress_info_controller)
        epochs_pbar, data_pbar = self._build_tqdm_contexts(trainer.train_dataloader)
        optimizer_with_param = [(Mock(), [Mock()])]

        assert trainer.disable_data_pbar == expected_disable_data_pbar  ### check setting disable_data_pbar flag

        with patch('model_compression_toolkit.gptq.keras.gptq_training.tqdm', side_effect=[epochs_pbar, data_pbar]) as tqdm_mock:
            trainer.micro_training_loop(Mock(), optimizer_with_param, 2, False)

        assert tqdm_mock.call_count == 2    ### check calling pbar count (epoch, data -> 2 times)
        assert tqdm_mock.call_args_list[1].kwargs['disable'] is expected_disable_data_pbar  ### chceck setting disable flag
