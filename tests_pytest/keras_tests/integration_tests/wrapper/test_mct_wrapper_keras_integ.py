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

"""
Test cases for MCTWrapper class from model_compression_toolkit.wrapper.mct_wrapper
"""
from unittest.mock import Mock, patch
from model_compression_toolkit.wrapper.mct_wrapper import MCTWrapper


class TestMCTWrapperIntegration:
    """
    Integration Tests for MCTWrapper Complete Workflows
    
    This test class focuses on testing the complete quantization and export
    workflows by testing the main quantize_and_export method with different
    configurations and scenarios.
    
    Test Categories:
        - PTQ Workflow: Complete Post-Training Quantization flow
        - GPTQ Mixed Precision: Gradient PTQ with mixed precision
        - LQ-PTQ TensorFlow: Low-bit quantization specific to TensorFlow
    """

    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._get_tpc')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._select_method')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._select_argname')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._setting_PTQ')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._export_model')
    def test_quantize_and_export_PTQ_flow(
            self, mock_export: Mock, mock_setting_ptq: Mock,
            mock_select_argname: Mock, mock_select_method: Mock,
            mock_get_tpc: Mock) -> None:
        """
        Test complete quantize_and_export workflow for Post-Training Quantization.
        
        This integration test verifies the complete PTQ workflow from input
        validation through model export. It mocks internal methods to focus
        on workflow coordination and method call sequences.
        
        Workflow Steps Tested:
            1. Input validation and initialization
            2. Parameter modification
            3. Method selection for framework and quantization type
            4. TPC (Target Platform Capabilities) configuration
            5. PTQ parameter setup
            6. Model quantization execution
            7. Model export
        
        Mocked Components:
            - _get_tpc: TPC configuration
            - _select_method: Framework-specific method selection
            - _setting_PTQ: PTQ parameter configuration
            - _export_model: Model export functionality
            - _post_training_quantization: Actual quantization process
        
        Verification Points:
            - Correct method call sequence
            - Proper parameter passing between methods
            - Expected return values (success flag and quantized model)
            - Instance state consistency after workflow completion
        """
        wrapper = MCTWrapper()
        
        # Setup mocks
        mock_float_model = Mock()
        mock_representative_dataset = Mock()
        mock_quantized_model = Mock()
        mock_info = Mock()
        
        # Mock the post_training_quantization method
        wrapper._post_training_quantization = Mock(
            return_value=(mock_quantized_model, mock_info))
        wrapper.export_model = Mock()
        wrapper._setting_PTQparam = mock_setting_ptq
        
        mock_setting_ptq.return_value = {'mock': 'params'}
        
        param_items = [('sdsp_version', '3.14')]  # SDSP version for TPC
        
        # Call the method
        success, result_model = wrapper.quantize_and_export(
            float_model=mock_float_model,
            framework='tensorflow',
            method='PTQ',
            use_mixed_precision=False,
            representative_dataset=mock_representative_dataset,
            param_items=param_items
        )
        
        # Verify the flow
        assert wrapper.float_model == mock_float_model
        assert wrapper.framework == 'tensorflow'
        assert wrapper.representative_dataset == mock_representative_dataset
        
        mock_get_tpc.assert_called_once_with()
        mock_select_method.assert_called_once_with()
        mock_select_argname.assert_called_once_with()
        mock_setting_ptq.assert_called_once()
        wrapper._post_training_quantization.assert_called_once_with(
            **{'mock': 'params'})
        mock_export.assert_called_once_with(mock_quantized_model)
        
        assert success is True
        assert result_model == mock_quantized_model

    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._get_tpc')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._select_method')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._select_argname')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._setting_GPTQ_mixed_precision')
    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._export_model')
    def test_quantize_and_export_GPTQ_mixed_precision_flow(
            self, mock_export: Mock, mock_setting_gptq_mixed_precision: Mock,
            mock_select_argname: Mock, mock_select_method: Mock,
            mock_get_tpc: Mock) -> None:
        """Test complete quantize_and_export flow for GPTQ with mixed_precision"""
        wrapper = MCTWrapper()
        
        # Setup mocks
        mock_float_model = Mock()
        mock_representative_dataset = Mock()
        mock_quantized_model = Mock()
        mock_info = Mock()
        
        wrapper._post_training_quantization = Mock(
            return_value=(mock_quantized_model, mock_info))
        wrapper.export_model = Mock()
        wrapper._setting_PTQparam = mock_setting_gptq_mixed_precision
        
        mock_setting_gptq_mixed_precision.return_value = {'mock': 'gptq_params'}
        
        # Call the method
        success, result_model = wrapper.quantize_and_export(
            float_model=mock_float_model,
            framework='tensorflow',
            method='GPTQ',
            use_mixed_precision=True,
            representative_dataset=mock_representative_dataset,
            param_items=[]
        )
        
        # Verify the flow
        mock_get_tpc.assert_called_once_with()
        mock_select_method.assert_called_once_with()
        mock_select_argname.assert_called_once_with()
        mock_setting_gptq_mixed_precision.assert_called_once()
        wrapper._post_training_quantization.assert_called_once_with(
            **{'mock': 'gptq_params'})
        mock_export.assert_called_once_with(mock_quantized_model)
        
        assert success is True
        assert result_model == mock_quantized_model

    @patch('model_compression_toolkit.wrapper.mct_wrapper.'
           'MCTWrapper._exec_lq_ptq')
    def test_quantize_and_export_LQPTQ(self, mock_exec_lq_ptq: Mock) -> None:
        """Test quantize_and_export flow for LQ-PTQ with TensorFlow"""
        wrapper = MCTWrapper()
        
        mock_float_model = Mock()
        mock_representative_dataset = Mock()
        mock_quantized_model = Mock()
        
        mock_exec_lq_ptq.return_value = mock_quantized_model
        
        # Call the method
        success, result_model = wrapper.quantize_and_export(
            float_model=mock_float_model,
            framework='tensorflow',
            method='LQPTQ',
            use_mixed_precision=False,
            representative_dataset=mock_representative_dataset,
            param_items=[]
        )
        
        # Verify the flow
        mock_exec_lq_ptq.assert_called_once()
        assert success is True
        assert result_model == mock_quantized_model