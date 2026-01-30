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
import pytest
from unittest.mock import Mock, patch
from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.wrapper.mct_wrapper import MCTWrapper


class TestMCTWrapper:
    """
    Unit Tests for MCTWrapper Core Functionality
    
    This test class focuses on testing individual methods and components
    of the MCTWrapper class in isolation. Each test uses mocking to avoid
    dependencies on external libraries and focuses on specific functionality.
    
    Test Categories:
        - Input Validation: Testing _initialize_and_validate success cases
        - Parameter Management: Testing _modify_params functionality
        - TPC Configuration: Testing _get_tpc with different TPC sources
        - Method Selection: Testing _select_method for different frameworks
        - Configuration Methods: Testing PTQ/GPTQ parameter generation
        - Export Functionality: Testing model export for different frameworks
    """

    def test_initialize_and_validate_valid_inputs(self) -> None:
        """
        Test _initialize_and_validate method with valid input parameters.
        
        This test verifies that the _initialize_and_validate method correctly
        initializes all wrapper instance attributes when provided with valid
        input parameters for all supported configurations.
        """
        wrapper = MCTWrapper()
        mock_model = Mock()
        mock_dataset = Mock()
        
        wrapper._initialize_and_validate(float_model=mock_model, representative_dataset=mock_dataset,
                                         framework='pytorch', method='PTQ', use_mixed_precision=False)

        assert wrapper.float_model == mock_model
        assert wrapper.framework == 'pytorch'
        assert wrapper.method == 'PTQ'
        assert wrapper.use_mixed_precision is False
        assert wrapper.representative_dataset == mock_dataset

    def test_modify_params(self) -> None:
        """
        Test _modify_params method with existing parameter keys.
        
        This test verifies that the _modify_params method correctly updates
        existing parameters in the wrapper's params dictionary when given
        valid parameter items.
        """
        wrapper = MCTWrapper()
        
        # Prepare test parameter items with existing keys
        param_items = [
            ['n_epochs', 10],  # Number of epochs
            ['learning_rate', 0.01],  # Learning rate
            ['fw_name', 'pytorch']  # Framework name
        ]
        
        # Call _modify_params to update existing parameters
        wrapper._modify_params(param_items)
        
        # Verify that parameters were updated correctly
        assert wrapper.params['n_epochs'] == 10
        assert wrapper.params['learning_rate'] == 0.01
        assert wrapper.params['fw_name'] == 'pytorch'

    def test_modify_params_non_existing_keys(self) -> None:
        """
        Test _modify_params method with non-existing parameter keys.
        
        This test verifies that the _modify_params method correctly handles
        parameter items that contain keys not present in the wrapper's default
        params dictionary. The method should ignore unknown keys and preserve
        all existing parameters unchanged.
        """
        wrapper = MCTWrapper()
        original_params = wrapper.params.copy()
        
        # Prepare test parameter items with non-existing keys
        param_items = [
            ['non_existing_key', 'value'],  # Non-existing key
            ['another_fake_key', 42]  # Another fake key
        ]
        
        # Call _modify_params
        wrapper._modify_params(param_items)
        
        # Check that original parameters are unchanged
        assert wrapper.params == original_params
        assert 'non_existing_key' not in wrapper.params
        assert 'another_fake_key' not in wrapper.params

    @patch('model_compression_toolkit.wrapper.mct_wrapper.mct.get_target_platform_capabilities_sdsp')
    def test_get_tpc(self, mock_mct_get_tpc_sdsp: Mock) -> None:
        """
        Test _get_tpc method.
        
        Verifies that the wrapper correctly calls
        mct.get_target_platform_capabilities_sdsp with expected parameters.
        
        Note: Patch targets mct.get_target_platform_capabilities_sdsp because
        MCTWrapper imports 'model_compression_toolkit as mct'.
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'pytorch'
        wrapper.params['sdsp_version'] = '3.14'
        mock_tpc = Mock()
        mock_mct_get_tpc_sdsp.return_value = mock_tpc
        
        wrapper._get_tpc()
        
        # Check if MCT get_target_platform_capabilities_sdsp was called correctly
        mock_mct_get_tpc_sdsp.assert_called_once_with(sdsp_version='3.14')
        assert wrapper.tpc == mock_tpc

    @patch('model_compression_toolkit.core.pytorch_resource_utilization_data')
    @patch('model_compression_toolkit.ptq.pytorch_post_training_quantization')
    @patch('model_compression_toolkit.gptq.'
           'pytorch_gradient_post_training_quantization')
    @patch('model_compression_toolkit.gptq.get_pytorch_gptq_config')
    @patch('model_compression_toolkit.exporter.pytorch_export_model')
    def test_select_method_PTQ(
            self, mock_pytorch_export: Mock, mock_pytorch_gptq_config: Mock,
            mock_pytorch_gptq: Mock, mock_pytorch_ptq: Mock,
            mock_pytorch_ru_data: Mock) -> None:
        """
        Test _select_method method for PyTorch framework with PTQ method.
        
        This test verifies that the _select_method method correctly configures
        all framework-specific function assignments when the wrapper is set to
        use PyTorch framework with Post-Training Quantization (PTQ).
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'pytorch'
        wrapper.method = 'PTQ'
        wrapper.use_mixed_precision = False
        
        wrapper._select_method()
        
        assert wrapper.params['fw_name'] == 'pytorch'
        assert wrapper.resource_utilization_data == mock_pytorch_ru_data
        assert wrapper._post_training_quantization == mock_pytorch_ptq
        assert wrapper.get_gptq_config == mock_pytorch_gptq_config
        assert wrapper.export_model == mock_pytorch_export

    @patch('model_compression_toolkit.core.pytorch_resource_utilization_data')
    @patch('model_compression_toolkit.ptq.pytorch_post_training_quantization')
    @patch('model_compression_toolkit.gptq.'
           'pytorch_gradient_post_training_quantization')
    @patch('model_compression_toolkit.gptq.get_pytorch_gptq_config')
    @patch('model_compression_toolkit.exporter.pytorch_export_model')
    def test_select_method_GPTQ(
            self, mock_pytorch_export: Mock, mock_pytorch_gptq_config: Mock,
            mock_pytorch_gptq: Mock, mock_pytorch_ptq: Mock,
            mock_pytorch_ru_data: Mock) -> None:
        """
        Test _select_method method for PyTorch framework with GPTQ method.
        
        This test verifies that the _select_method method correctly configures
        all framework-specific function assignments when the wrapper is set to
        use PyTorch framework with Gradient Post-Training Quantization.
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'pytorch'
        wrapper.method = 'GPTQ'
        wrapper.use_mixed_precision = False
        
        wrapper._select_method()
        
        assert wrapper.params['fw_name'] == 'pytorch'
        assert wrapper.resource_utilization_data == mock_pytorch_ru_data
        assert wrapper._post_training_quantization == mock_pytorch_gptq
        assert wrapper.get_gptq_config == mock_pytorch_gptq_config
        assert wrapper.export_model == mock_pytorch_export

    def test_select_argname(self) -> None:
        """
        Test select_argname method for PyTorch framework.
        
        This test verifies that the select_argname method correctly sets
        argument names specific to PyTorch framework for parameter
        dictionaries used in quantization methods.
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'pytorch'
        
        wrapper._select_argname()
        
        # PyTorch should use IN_MODULE for argname_in_module
        assert wrapper.argname_in_module == 'in_module'
        # PyTorch should use MODEL for argname_model
        assert wrapper.argname_model == 'model'

    @patch('model_compression_toolkit.core.ResourceUtilization')
    @patch('model_compression_toolkit.core.CoreConfig')
    @patch('model_compression_toolkit.core.MixedPrecisionQuantizationConfig')
    def test_setting_PTQ_mixed_precision(
            self, mock_mixed_precision_config: Mock, mock_core_config: Mock,
            mock_resource_util: Mock) -> None:
        """
        Test _setting_PTQ_mixed_precision method for Mixed Precision PTQ configuration.
        
        This test verifies that the _setting_PTQ_mixed_precision method correctly configures
        mixed precision Post-Training Quantization parameters by properly setting
        up configuration objects and resource utilization constraints.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'pytorch'
        
        # Mock resource utilization data
        mock_ru_data = Mock()
        mock_ru_data.weights_memory = 1000
        wrapper.resource_utilization_data = Mock(return_value=mock_ru_data)
        
        # Mock config objects
        mock_mp_config_instance = Mock()
        mock_mixed_precision_config.return_value = mock_mp_config_instance
        mock_ptq_config_instance = Mock()
        mock_core_config.return_value = mock_ptq_config_instance
        mock_resource_util_instance = Mock()
        mock_resource_util.return_value = mock_resource_util_instance
        
        wrapper._select_argname()
        result = wrapper._setting_PTQ_mixed_precision()
        
        # Verify the method calls
        mock_mixed_precision_config.assert_called_with(
            distance_weighting_method=None,
            num_of_images=32,
            use_hessian_based_scores=False
        )
        # Verify core_config was called (exact parameters may vary due to mock setup)
        mock_core_config.assert_called_once()
        mock_resource_util.assert_called_with(750.0)  # 1000 * 0.75
        
        # Check result structure
        assert 'in_module' in result  # PyTorch uses in_module
        assert 'representative_data_gen' in result
        assert 'target_resource_utilization' in result
        assert 'core_config' in result
        assert 'target_platform_capabilities' in result

    @patch('model_compression_toolkit.core.QuantizationConfig')
    @patch('model_compression_toolkit.core.CoreConfig')
    def test_setting_PTQ(self, mock_core_config: Mock,
                         mock_quant_config: Mock) -> None:
        """
        Test _setting_PTQ method for standard Post-Training Quantization.
        
        This test verifies that the _setting_PTQ method correctly configures
        standard Post-Training Quantization parameters without mixed precision,
        focusing on fixed-precision quantization with comprehensive error
        minimization and optimization techniques.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'pytorch'
        
        # Mock config objects
        mock_quant_config_instance = Mock()
        mock_quant_config.return_value = mock_quant_config_instance
        mock_ptq_config_instance = Mock()
        mock_core_config.return_value = mock_ptq_config_instance
        
        wrapper._select_argname()
        result = wrapper._setting_PTQ()
        
        # Verify the method calls
        mock_quant_config.assert_called_with(
            activation_error_method=QuantizationErrorMethod.MSE,
            weights_bias_correction=True,
            z_threshold=float('inf'),
            linear_collapsing=True,
            residual_collapsing=True
        )
        mock_core_config.assert_called_with(
            quantization_config=mock_quant_config_instance)
        
        # Check result structure for PyTorch
        assert 'in_module' in result  # PyTorch uses in_module
        assert result['target_resource_utilization'] is None

    @patch('model_compression_toolkit.core.QuantizationConfig')
    @patch('model_compression_toolkit.core.MixedPrecisionQuantizationConfig')
    @patch('model_compression_toolkit.core.CoreConfig')
    @patch('model_compression_toolkit.core.ResourceUtilization')
    def test_setting_GPTQ_mixed_precision(
            self, mock_resource_util: Mock, mock_core_config: Mock,
            mock_mixed_precision_config: Mock,
            mock_quant_config: Mock) -> None:
        """
        Test _setting_GPTQ_mixed_precision method for Mixed Precision GPTQ configuration.
        
        This test verifies that the _setting_GPTQ_mixed_precision method correctly
        configures mixed precision Gradient Post-Training Quantization
        parameters with proper configuration objects and resource utilization.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'pytorch'
        wrapper.get_gptq_config = Mock(return_value=Mock())
        
        # Mock resource utilization data
        mock_ru_data = Mock()
        mock_ru_data.weights_memory = 1000
        wrapper.resource_utilization_data = Mock(return_value=mock_ru_data)
        
        # Mock config objects
        mock_mp_config_instance = Mock()
        mock_mixed_precision_config.return_value = mock_mp_config_instance
        mock_quant_config_instance = Mock()
        mock_quant_config.return_value = mock_quant_config_instance
        mock_core_config_instance = Mock()
        mock_core_config.return_value = mock_core_config_instance
        mock_resource_util_instance = Mock()
        mock_resource_util.return_value = mock_resource_util_instance

        wrapper._select_argname()
        result = wrapper._setting_GPTQ_mixed_precision()
        
        # Verify the method calls
        mock_mixed_precision_config.assert_called_with(
            distance_weighting_method=None,
            num_of_images=32,
            use_hessian_based_scores=False
        )
        # Verify quant_config was called (with standard quantization parameters)
        assert mock_quant_config.called
        mock_resource_util.assert_called_with(750.0)  # 1000 * 0.75
        
        # Check that PyTorch-specific parameter mapping is applied
        assert 'model' in result
        assert result['model'] == wrapper.float_model
        assert 'representative_data_gen' in result
        assert 'target_resource_utilization' in result
        assert 'gptq_config' in result
        assert 'core_config' in result
        assert 'target_platform_capabilities' in result

    def test_setting_GPTQ(self) -> None:
        """
        Test _setting_GPTQ method for PyTorch framework configuration.
        
        This test verifies that the _setting_GPTQ method correctly configures
        Gradient Post-Training Quantization (GPTQ) parameters specifically for
        PyTorch framework, ensuring proper parameter mapping and framework-
        specific API compatibility.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'pytorch'
        wrapper.get_gptq_config = Mock(return_value=Mock())

        wrapper._select_argname()
        result = wrapper._setting_GPTQ()
        
        # Check that PyTorch uses 'model' parameter
        assert 'model' in result
        assert result['model'] == wrapper.float_model

    def test_export_model(self) -> None:
        """
        Test _export_model method for PyTorch framework export functionality.
        
        This test verifies that the _export_model method correctly exports
        quantized PyTorch models to ONNX format with appropriate parameters
        and framework-specific configurations.
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'pytorch'
        wrapper.params['save_model_path'] = './test_model.onnx'
        wrapper.representative_dataset = Mock()
        wrapper.export_model = Mock()
        
        mock_quantized_model = Mock()
        
        wrapper._export_model(mock_quantized_model)
        
        # Verify export function was called with correct parameters
        wrapper.export_model.assert_called_once()
        call_args = wrapper.export_model.call_args[1]  # Get keyword arguments
        assert call_args['model'] == mock_quantized_model
        assert call_args['save_model_path'] == './test_model.onnx'


class TestMCTWrapperErrorHandling:
    """
    Error Handling and Edge Case Tests for MCTWrapper
    
    This test class focuses on testing error conditions, invalid inputs,
    and edge cases to ensure robust error handling throughout the MCTWrapper
    functionality.
    
    Error Categories Tested:
        - Invalid Method Parameters: Unsupported quantization methods
        - Framework Compatibility: Invalid framework combinations
        - Method Restrictions: Method-framework incompatibilities
    
    Testing Approach:
        - Uses pytest.raises to verify expected exceptions
        - Tests error message content for clarity
        - Covers boundary conditions and invalid input combinations
        - Ensures proper exception propagation from internal methods
    
    Key Validation Points:
        - Method support validation (PTQ, GPTQ, LQPTQ)
        - Framework support validation (TensorFlow, PyTorch)
        - Cross-compatibility validation (LQ-PTQ only with TensorFlow)
    """

    def test_quantize_and_export_unsupported_method(self) -> None:
        """Test quantize_and_export with unsupported method"""
        wrapper = MCTWrapper()
        
        with pytest.raises(Exception) as exc_info:
            wrapper.quantize_and_export(
                float_model=Mock(),
                representative_dataset=Mock(),
                framework='pytorch',
                method='UNSUPPORTED_METHOD',
                use_mixed_precision=False,
                param_items=[]
            )
        
        expected_msg = "Only PTQ, GPTQ and LQPTQ are supported now"
        assert expected_msg in str(exc_info.value)

    def test_quantize_and_export_LQPTQ_with_pytorch(self) -> None:
        """Test quantize_and_export with LQPTQ method and PyTorch framework"""
        wrapper = MCTWrapper()
        
        with pytest.raises(Exception) as exc_info:
            wrapper.quantize_and_export(
                float_model=Mock(),
                representative_dataset=Mock(),
                framework='pytorch',
                method='LQPTQ',
                use_mixed_precision=False,
                param_items=[]
            )
        
        expected_msg = "LQ-PTQ is only supported with tensorflow now"
        assert expected_msg in str(exc_info.value)

    def test_quantize_and_export_unsupported_framework(self) -> None:
        """Test quantize_and_export with unsupported framework"""
        wrapper = MCTWrapper()
        
        with pytest.raises(Exception) as exc_info:
            wrapper.quantize_and_export(
                float_model=Mock(),
                representative_dataset=Mock(),
                framework='unsupported',
                method='PTQ',
                use_mixed_precision=False,
                param_items=[]
            )
        
        expected_msg = "Only tensorflow and pytorch are supported now"
        assert expected_msg in str(exc_info.value)