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

from typing import Dict, Any, List, Optional, Tuple
import model_compression_toolkit as mct
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.wrapper.constants import (
    FW_NAME, SDSP_VERSION, ACTIVATION_ERROR_METHOD, WEIGHTS_BIAS_CORRECTION,
    Z_THRESHOLD, LINEAR_COLLAPSING, RESIDUAL_COLLAPSING, 
    DISTANCE_WEIGHTING_METHOD, NUM_OF_IMAGES, 
    USE_HESSIAN_BASED_SCORES, WEIGHTS_COMPRESSION_RATIO,
    IN_MODEL, REPRESENTATIVE_DATA_GEN, CORE_CONFIG, TARGET_PLATFORM_CAPABILITIES,
    TARGET_RESOURCE_UTILIZATION, IN_MODULE, GPTQ_CONFIG, MODEL,
    N_EPOCHS, OPTIMIZER, LEARNING_RATE, CONVERTER_VER, SAVE_MODEL_PATH, DEFAULT_COMPRESSION_RATIO
)


class MCTWrapper:
    """
    Wrapper class for Model Compression Toolkit (MCT) quantization and export.

    This class provides a unified interface for various neural network
    quantization methods including Post-Training Quantization (PTQ), Gradient
    Post-Training Quantization (GPTQ).
    It supports both TensorFlow and PyTorch frameworks with optional
    mixed-precision quantization.

    The wrapper manages the complete quantization pipeline from model input to
    quantized model export, handling framework-specific configurations and
    Target Platform Capabilities (TPC) setup.
    """

    def __init__(self):
        self.params: Dict[str, Any] = {
            # TPC
            FW_NAME: 'pytorch',
            SDSP_VERSION: '3.14',

            # QuantizationConfig
            ACTIVATION_ERROR_METHOD: mct.core.QuantizationErrorMethod.MSE,
            WEIGHTS_BIAS_CORRECTION: True,
            Z_THRESHOLD: float('inf'),
            LINEAR_COLLAPSING: True,
            RESIDUAL_COLLAPSING: True,

            # MixedPrecisionQuantizationConfig
            DISTANCE_WEIGHTING_METHOD: None,
            NUM_OF_IMAGES: 32,
            USE_HESSIAN_BASED_SCORES: False,

            # ResourceUtilization
            WEIGHTS_COMPRESSION_RATIO: DEFAULT_COMPRESSION_RATIO,

            # GradientPTQConfig
            N_EPOCHS: 5,
            OPTIMIZER: None,

            # low_bit_quantizer_ptq
            LEARNING_RATE: 0.001,
            CONVERTER_VER: 'v3.14',

            # Export
            SAVE_MODEL_PATH: './qmodel.onnx'
        }

    def _initialize_and_validate(self, float_model: Any,
                                 representative_dataset: Optional[Any],
                                 framework: str,
                                 method: str,
                                 use_mixed_precision: bool
                                 ) -> None:
        """
        Validate inputs and initialize parameters.

        Args:
            float_model: The float model to be quantized.
            representative_dataset (Callable, np.array, tf.Tensor): Representative dataset for calibration.
            framework (str): Target framework ('tensorflow', 'pytorch').
            method (str): Quantization method ('PTQ', 'GPTQ', 'LQPTQ').
            use_mixed_precision (bool): Whether to use mixed-precision quantization.

        Raises:
            Exception: If method or framework is not supported.
        """
        # error check --------------------------
        if method not in ['PTQ', 'GPTQ', 'LQPTQ']:
            raise Exception("Only PTQ, GPTQ and LQPTQ are supported now")
        if method == 'LQPTQ' and framework != 'tensorflow':
            raise Exception("LQ-PTQ is only supported with tensorflow now") 
        if framework not in ['tensorflow', 'pytorch']:
            raise Exception("Only tensorflow and pytorch are supported now")        
      
        # set parameters --------------------------
        self.float_model = float_model
        self.representative_dataset = representative_dataset
        self.framework = framework
        self.method = method
        self.use_mixed_precision = use_mixed_precision

        # Keep only the parameters you need for the quantization mode
        if method == 'PTQ':
            if not use_mixed_precision:
                allowed_keys = [FW_NAME, SDSP_VERSION, ACTIVATION_ERROR_METHOD, WEIGHTS_BIAS_CORRECTION,
                                Z_THRESHOLD, LINEAR_COLLAPSING, RESIDUAL_COLLAPSING,
                                SAVE_MODEL_PATH]
            else:
                allowed_keys = [FW_NAME, SDSP_VERSION, ACTIVATION_ERROR_METHOD, WEIGHTS_BIAS_CORRECTION,
                                Z_THRESHOLD, LINEAR_COLLAPSING, RESIDUAL_COLLAPSING,
                                DISTANCE_WEIGHTING_METHOD, NUM_OF_IMAGES, USE_HESSIAN_BASED_SCORES,
                                WEIGHTS_COMPRESSION_RATIO, SAVE_MODEL_PATH]
        else:
            if not use_mixed_precision:
                allowed_keys = [FW_NAME, SDSP_VERSION, ACTIVATION_ERROR_METHOD, WEIGHTS_BIAS_CORRECTION, 
                                Z_THRESHOLD, LINEAR_COLLAPSING, RESIDUAL_COLLAPSING,
                                N_EPOCHS, OPTIMIZER, SAVE_MODEL_PATH]
            else:
                allowed_keys = [FW_NAME, SDSP_VERSION, ACTIVATION_ERROR_METHOD, WEIGHTS_BIAS_CORRECTION, 
                                Z_THRESHOLD, LINEAR_COLLAPSING, RESIDUAL_COLLAPSING,
                                WEIGHTS_COMPRESSION_RATIO, N_EPOCHS, OPTIMIZER, DISTANCE_WEIGHTING_METHOD,
                                NUM_OF_IMAGES, USE_HESSIAN_BASED_SCORES,
                                SAVE_MODEL_PATH]
                     
        self.params = { k: v for k, v in self.params.items() if k in allowed_keys }

        if self.framework == 'tensorflow':
            self.params[SAVE_MODEL_PATH] = './qmodel.keras'

    def _modify_params(self, param_items: List[List[Any]]) -> None:
        """
        Update the internal parameter dictionary with values from param_items.

        Args:
            param_items (list): List of lists [[key, value], ...].
                If key exists in self.params, updates its value.
                Non-existing keys are ignored with a warning.

        Note:
            Only parameters that exist in the default parameter dictionary
            will be updated. Unknown parameters are silently ignored.
        """
        if param_items is None:
            return
        
        for key, value in param_items:
            if key in self.params:
                # Update parameter value if key exists in default parameters
                self.params[key] = value
            else:
                Logger.warning(f"The key '{key}' is not found in the default "
                               f"parameters and will be ignored.")

    def _select_method(self) -> None:
        """
        Select and set appropriate quantization, export, and config methods.

        Configures framework-specific methods based on the backend
        (Keras/PyTorch) and quantization method (PTQ/GPTQ). Also sets up
        method-specific parameter configuration functions.

        Note:
            This method dynamically assigns methods to instance attributes
            based on self.framework and self.method values.
        """
        if self.framework == 'tensorflow':
            # Set TensorFlow/Keras specific methods and parameters
            self.params[FW_NAME] = 'tensorflow'
            self.resource_utilization_data = mct.core.keras_resource_utilization_data
            self.get_gptq_config = mct.gptq.get_keras_gptq_config
            self.export_model = mct.exporter.keras_export_model
        elif self.framework == 'pytorch':
            # Set PyTorch specific methods and parameters
            self.params[FW_NAME] = 'pytorch'
            self.resource_utilization_data = mct.core.pytorch_resource_utilization_data
            self.get_gptq_config = mct.gptq.get_pytorch_gptq_config
            self.export_model = mct.exporter.pytorch_export_model
        else:
            raise Exception("Only tensorflow and pytorch are supported now")

        if self.method == 'PTQ':
            # Set Post-Training Quantization methods
            if self.framework == 'tensorflow':
                self._post_training_quantization = mct.ptq.keras_post_training_quantization
            elif self.framework == 'pytorch':
                self._post_training_quantization = mct.ptq.pytorch_post_training_quantization

            if self.use_mixed_precision:
                # Use mixed precision PTQ parameter configuration
                self._setting_PTQparam = self._setting_PTQ_mixed_precision
            else:
                # Use standard PTQ parameter configuration
                self._setting_PTQparam = self._setting_PTQ

        elif self.method == 'GPTQ':
            # Set Gradient Post-Training Quantization methods
            if self.framework == 'tensorflow':
                self._post_training_quantization = mct.gptq.keras_gradient_post_training_quantization
            elif self.framework == 'pytorch':
                self._post_training_quantization = mct.gptq.pytorch_gradient_post_training_quantization

            if self.use_mixed_precision:
                # Use mixed precision GPTQ parameter configuration
                self._setting_PTQparam = self._setting_GPTQ_mixed_precision
            else:
                # Use standard GPTQ parameter configuration
                self._setting_PTQparam = self._setting_GPTQ

    def _select_argname(self) -> None:
        """
        Select argument names based on the framework and method.
        
        This method configures framework-specific parameter names used in 
        quantization method calls. Different frameworks (TensorFlow/PyTorch) 
        and methods (PTQ/GPTQ) require different parameter names for the same 
        conceptual arguments.
        
        Sets:
            argname_in_module: Parameter name for model input in PTQ methods
                - TensorFlow: 'in_model' 
                - PyTorch: 'in_module'
            argname_model: Parameter name for model input in GPTQ methods
                - TensorFlow: 'in_model'
                - PyTorch: 'model'
        
        Note:
            This method must be called after _select_method() and before 
            calling any _setting_* methods that use these parameter names.
        """
        if self.framework == 'tensorflow':
            self.argname_in_module = IN_MODEL
        elif self.framework == 'pytorch':
            self.argname_in_module = IN_MODULE

        if self.framework == 'tensorflow':
            self.argname_model = IN_MODEL
        elif self.framework == 'pytorch':
            self.argname_model = MODEL

    def _get_tpc(self) -> None:
        """
        Configure Target Platform Capabilities (TPC).

        Sets up TPC configuration for the target platform.

        Note:
            This method sets self.tpc attribute with the configured TPC object.
        """
        # Get default TPC for the framework
        params_TPC = {
            SDSP_VERSION: self.params[SDSP_VERSION]
        }
        self.tpc = mct.get_target_platform_capabilities_sdsp(**params_TPC)

    def _setting_PTQ_mixed_precision(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for mixed-precision PTQ.

        Returns:
            dict: Parameter dictionary for PTQ.
        """
        params_QCfg = {
            ACTIVATION_ERROR_METHOD: self.params[ACTIVATION_ERROR_METHOD],
            WEIGHTS_BIAS_CORRECTION: self.params[WEIGHTS_BIAS_CORRECTION],
            Z_THRESHOLD: self.params[Z_THRESHOLD],
            LINEAR_COLLAPSING: self.params[LINEAR_COLLAPSING],
            RESIDUAL_COLLAPSING: self.params[RESIDUAL_COLLAPSING]
        }
        q_config = mct.core.QuantizationConfig(**params_QCfg)
        
        params_MPCfg = {
            DISTANCE_WEIGHTING_METHOD: self.params[DISTANCE_WEIGHTING_METHOD],
            NUM_OF_IMAGES: self.params[NUM_OF_IMAGES],
            USE_HESSIAN_BASED_SCORES: self.params[USE_HESSIAN_BASED_SCORES]
        }
        mixed_precision_config = mct.core.MixedPrecisionQuantizationConfig(**params_MPCfg)

        core_config = mct.core.CoreConfig(quantization_config=q_config, 
                                          mixed_precision_config=mixed_precision_config)
       
        params_RUDCfg = {
            IN_MODEL: self.float_model,
            REPRESENTATIVE_DATA_GEN: self.representative_dataset,
            CORE_CONFIG: core_config,
            TARGET_PLATFORM_CAPABILITIES: self.tpc
        }
        ru_data = self.resource_utilization_data(**params_RUDCfg)
        weights_compression_ratio = self.params[WEIGHTS_COMPRESSION_RATIO]
        resource_utilization = mct.core.ResourceUtilization(
            ru_data.weights_memory * weights_compression_ratio)

        params_PTQ = {
            self.argname_in_module: self.float_model,
            REPRESENTATIVE_DATA_GEN: self.representative_dataset,
            TARGET_RESOURCE_UTILIZATION: resource_utilization,
            CORE_CONFIG: core_config,
            TARGET_PLATFORM_CAPABILITIES: self.tpc
        }
        return params_PTQ

    def _setting_PTQ(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for PTQ.

        Returns:
            dict: Parameter dictionary for PTQ.
        """
        params_QCfg = {
            ACTIVATION_ERROR_METHOD: self.params[ACTIVATION_ERROR_METHOD],
            WEIGHTS_BIAS_CORRECTION: self.params[WEIGHTS_BIAS_CORRECTION],
            Z_THRESHOLD: self.params[Z_THRESHOLD],
            LINEAR_COLLAPSING: self.params[LINEAR_COLLAPSING],
            RESIDUAL_COLLAPSING: self.params[RESIDUAL_COLLAPSING]
        }
        q_config = mct.core.QuantizationConfig(**params_QCfg)
        core_config = mct.core.CoreConfig(quantization_config=q_config)
        resource_utilization = None

        params_PTQ = {
            self.argname_in_module: self.float_model,
            REPRESENTATIVE_DATA_GEN: self.representative_dataset,
            TARGET_RESOURCE_UTILIZATION: resource_utilization,
            CORE_CONFIG: core_config,
            TARGET_PLATFORM_CAPABILITIES: self.tpc
        }
        return params_PTQ

    def _setting_GPTQ_mixed_precision(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for mixed-precision GPTQ.

        Returns:
            dict: Parameter dictionary for GPTQ.
        """
        params_QCfg = {
            ACTIVATION_ERROR_METHOD: self.params[ACTIVATION_ERROR_METHOD],
            WEIGHTS_BIAS_CORRECTION: self.params[WEIGHTS_BIAS_CORRECTION],
            Z_THRESHOLD: self.params[Z_THRESHOLD],
            LINEAR_COLLAPSING: self.params[LINEAR_COLLAPSING],
            RESIDUAL_COLLAPSING: self.params[RESIDUAL_COLLAPSING]
        }
        q_config = mct.core.QuantizationConfig(**params_QCfg)        
         
        params_GPTQCfg = {
            N_EPOCHS: self.params[N_EPOCHS],
            OPTIMIZER: self.params[OPTIMIZER]
        }
        gptq_config = self.get_gptq_config(**params_GPTQCfg)

        params_MPCfg = {
            DISTANCE_WEIGHTING_METHOD: self.params[DISTANCE_WEIGHTING_METHOD],
            NUM_OF_IMAGES: self.params[NUM_OF_IMAGES],
            USE_HESSIAN_BASED_SCORES: self.params[USE_HESSIAN_BASED_SCORES],
        }
        mixed_precision_config = mct.core.MixedPrecisionQuantizationConfig(**params_MPCfg)

        core_config = mct.core.CoreConfig(quantization_config=q_config,
                                          mixed_precision_config=mixed_precision_config)

        params_RUDCfg = {
            IN_MODEL: self.float_model,
            REPRESENTATIVE_DATA_GEN: self.representative_dataset,
            CORE_CONFIG: core_config,
            TARGET_PLATFORM_CAPABILITIES: self.tpc
        }
        ru_data = self.resource_utilization_data(**params_RUDCfg)
        weights_compression_ratio = self.params[WEIGHTS_COMPRESSION_RATIO]
        resource_utilization = mct.core.ResourceUtilization(
            ru_data.weights_memory * weights_compression_ratio)

        params_GPTQ = {
            self.argname_model: self.float_model,
            REPRESENTATIVE_DATA_GEN: self.representative_dataset,
            TARGET_RESOURCE_UTILIZATION: resource_utilization,
            GPTQ_CONFIG: gptq_config,
            CORE_CONFIG: core_config,
            TARGET_PLATFORM_CAPABILITIES: self.tpc
        }
        return params_GPTQ

    def _setting_GPTQ(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for GPTQ.

        Returns:
            dict: Parameter dictionary for GPTQ.
        """
        params_QCfg = {
            ACTIVATION_ERROR_METHOD: self.params[ACTIVATION_ERROR_METHOD],
            WEIGHTS_BIAS_CORRECTION: self.params[WEIGHTS_BIAS_CORRECTION],
            Z_THRESHOLD: self.params[Z_THRESHOLD],
            LINEAR_COLLAPSING: self.params[LINEAR_COLLAPSING],
            RESIDUAL_COLLAPSING: self.params[RESIDUAL_COLLAPSING]
        }
        q_config = mct.core.QuantizationConfig(**params_QCfg)
        core_config = mct.core.CoreConfig(quantization_config=q_config)

        params_GPTQCfg = {
            N_EPOCHS: self.params[N_EPOCHS],
            OPTIMIZER: self.params[OPTIMIZER]
        }
        gptq_config = self.get_gptq_config(**params_GPTQCfg)

        params_GPTQ = {
            self.argname_model: self.float_model,
            REPRESENTATIVE_DATA_GEN: self.representative_dataset,
            GPTQ_CONFIG: gptq_config,
            CORE_CONFIG: core_config,
            TARGET_PLATFORM_CAPABILITIES: self.tpc
        }
        return params_GPTQ

    def _exec_lq_ptq(self) -> Any:
        """
        Execute Low-bit Quantization Post-Training Quantization (LQ-PTQ).

        Performs quantization using the low_bit_quantizer_ptq method with
        the configured parameters and representative dataset.

        Returns:
            The quantized model object.

        Note:
            This method requires the lq_ptq module to be imported.
        """
        # Placeholder implementation - replace with actual lq_ptq call
        raise NotImplementedError(
            "LQ-PTQ functionality requires lq_ptq module to be imported")

    def _export_model(self, quantized_model: Any) -> None:
        """
        Export the quantized model using appropriate export function.

        Configures export parameters based on the framework and exports
        the quantized model to the specified path.

        Args:
            quantized_model: The quantized model to export.

        Note:
            Export format is framework-specific: Keras for TensorFlow,
            ONNX for PyTorch.
        """
        if self.framework == 'tensorflow':
            params_export = {
                'model': quantized_model,
                'save_model_path': self.params['save_model_path'],
            }
        elif self.framework == 'pytorch':
            params_export = {
                'model': quantized_model,
                'save_model_path': self.params['save_model_path'],
                'repr_dataset': self.representative_dataset
            }
        self.export_model(**params_export)

    def quantize_and_export(self, float_model: Any,
                            representative_dataset: Any,
                            framework: str = 'pytorch',
                            method: str = 'PTQ',
                            use_mixed_precision: bool = False,
                            param_items: Optional[List[List[Any]]] = None
                            ) -> Tuple[bool, Any]:
        """
        Main function to perform model quantization and export.

        Args:
            float_model: The float model to be quantized.
            representative_dataset (Callable, np.array, tf.Tensor):
                Representative dataset for calibration.
            framework (str): 'tensorflow' or 'pytorch'.
                Default: 'pytorch'
            method (str): Quantization method, e.g., 'PTQ' or 'GPTQ'.
                Default: 'PTQ'
            use_mixed_precision (bool): Whether to use mixed-precision
                quantization. Default: False
            param_items (list): List of parameter settings.
                [[key,value],...]. Default: None

        Returns:
            tuple (quantization success flag, quantized model)
            
        Examples:

            Import MCT

            >>> import model_compression_toolkit as mct
            
            Prepare the float model and dataset
            
            >>> float_model = ...
            >>> representative_dataset = ...
          
            Create an instance of the MCTWrapper

            >>> wrapper = mct.MCTWrapper()

            Set framework, method, and other parameters

            >>> framework = 'tensorflow'
            >>> method = 'PTQ'
            >>> use_mixed_precision = False

            Set parameters if needed

            >>> param_items = [[key, value]...]

            Quantize and export the model

            >>> flag, quantized_model = wrapper.quantize_and_export(
            ...     float_model=float_model,
            ...     representative_dataset=representative_dataset,
            ...     framework=framework,
            ...     method=method,
            ...     use_mixed_precision=use_mixed_precision,
            ...     param_items=param_items
            ... )

        **Parameters**
    
        Initialize MCTWrapper with default parameters

        Users can update the following parameters in param_items.
    
        .. note::
           The low priority variable can be left at its default value, so there is no need to specify it.
           Specify it as necessary, for example, if you receive a warning from the `XQuant Extension Tool <https://sonysemiconductorsolutions.github.io/mct-model-optimization/guidelines/XQuant_Extension_Tool.html>`_.
    
        PTQ
    
        .. csv-table::
           :header: "Parameter Key", "Default Value", "Description"
           :widths: 30, 30, 40
    
           "sdsp_version", "'3.14'", "By specifying the SDSP converter version, you can select the `optimal quantization settings <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/modules/target_platform_capabilities.html#ug-target-platform-capabilities>`_ for IMX500."
           "save_model_path", "'./qmodel.keras' / './qmodel.onnx'", "Path to save quantized model (Keras/Pytorch)"
           "activation_error_method", "mct.core.QuantizationErrorMethod.MSE", "Activation quantization error method **(low priority)**"
           "weights_bias_correction", "True", "Enable weights bias correction **(low priority)**"
           "z_threshold", "float('inf')", "Z-threshold for quantization **(low priority)**"
           "linear_collapsing", "True", "Enable linear layer collapsing **(low priority)**"
           "residual_collapsing", "True", "Enable residual connection collapsing **(low priority)**"
    
        PTQ, mixed_precision
    
        .. csv-table::
           :header: "Parameter Key", "Default Value", "Description"
           :widths: 30, 30, 40
    
           "sdsp_version", "'3.14'", "By specifying the SDSP converter version, you can select the `optimal quantization settings <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/modules/target_platform_capabilities.html#ug-target-platform-capabilities>`_ for IMX500."
           "save_model_path", "'./qmodel.keras' / './qmodel.onnx'", "Path to save quantized model (Keras/Pytorch)"
           "num_of_images", "32", "Number of images for mixed precision"
           "weights_compression_ratio", "0.75", "Weights compression ratio for mixed precision for resource util (0.0～1.0)"
           "activation_error_method", "mct.core.QuantizationErrorMethod.MSE", "Activation quantization error method **(low priority)**"
           "weights_bias_correction", "True", "Enable weights bias correction **(low priority)**"
           "z_threshold", "float('inf')", "Z-threshold for quantization **(low priority)**"
           "linear_collapsing", "True", "Enable linear layer collapsing **(low priority)**"
           "residual_collapsing", "True", "Enable residual connection collapsing **(low priority)**"
           "distance_weighting_method", "default of `MixedPrecisionQuantizationConfig <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/classes/MixedPrecisionQuantizationConfig.html#mpdistanceweighting>`_", "Distance weighting method for mixed precision **(low priority)**"
           "use_hessian_based_scores", "False", "Use Hessian-based scores for mixed precision **(low priority)**"
    
        GPTQ
    
        .. csv-table::
           :header: "Parameter Key", "Default Value", "Description"
           :widths: 30, 30, 40
    
           "sdsp_version", "'3.14'", "By specifying the SDSP converter version, you can select the `optimal quantization settings <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/modules/target_platform_capabilities.html#ug-target-platform-capabilities>`_ for IMX500."
           "save_model_path", "'./qmodel.keras' / './qmodel.onnx'", "Path to save quantized model (Keras/Pytorch)"
           "n_epochs", "5", "Number of training epochs for GPTQ"
           "activation_error_method", "mct.core.QuantizationErrorMethod.MSE", "Activation quantization error method **(low priority)**"
           "weights_bias_correction", "True", "Enable weights bias correction **(low priority)**"
           "z_threshold", "float('inf')", "Z-threshold for quantization **(low priority)**"
           "linear_collapsing", "True", "Enable linear layer collapsing **(low priority)**"
           "residual_collapsing", "True", "Enable residual connection collapsing **(low priority)**"
           "optimizer", "default of `get_keras_gptq_config <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/methods/get_keras_gptq_config.html#model_compression_toolkit.gptq.get_keras_gptq_config>`_ or `get_pytorch_gptq_config <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/methods/get_pytroch_gptq_config.html#model_compression_toolkit.gptq.get_pytorch_gptq_config>`_", "Optimizer for GPTQ **(low priority)**"
    
        GPTQ, mixed_precision
    
        .. csv-table::
           :header: "Parameter Key", "Default Value", "Description"
           :widths: 30, 30, 40
    
           "sdsp_version", "'3.14'", "By specifying the SDSP converter version, you can select the `optimal quantization settings <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/modules/target_platform_capabilities.html#ug-target-platform-capabilities>`_ for IMX500."
           "save_model_path", "'./qmodel.keras' / './qmodel.onnx'", "Path to save quantized model (Keras/Pytorch)"
           "num_of_images", "32", "Number of images for mixed precision"
           "weights_compression_ratio", "0.75", "Weights compression ratio for mixed precision for resource util (0.0～1.0)"
           "n_epochs", "5", "Number of training epochs for GPTQ"
           "activation_error_method", "mct.core.QuantizationErrorMethod.MSE", "Activation quantization error method **(low priority)**"
           "weights_bias_correction", "True", "Enable weights bias correction **(low priority)**"
           "z_threshold", "float('inf')", "Z-threshold for quantization **(low priority)**"
           "linear_collapsing", "True", "Enable linear layer collapsing **(low priority)**"
           "residual_collapsing", "True", "Enable residual connection collapsing **(low priority)**"
           "distance_weighting_method", "default of `MixedPrecisionQuantizationConfig <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/classes/MixedPrecisionQuantizationConfig.html#mpdistanceweighting>`_", "Distance weighting method for mixed precision **(low priority)**"
           "use_hessian_based_scores", "False", "Use Hessian-based scores for mixed precision **(low priority)**"
           "optimizer", "default of `get_keras_gptq_config <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/methods/get_keras_gptq_config.html#model_compression_toolkit.gptq.get_keras_gptq_config>`_ or `get_pytorch_gptq_config <https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/methods/get_pytroch_gptq_config.html#model_compression_toolkit.gptq.get_pytorch_gptq_config>`_", "Optimizer for GPTQ **(low priority)**"

        """
        try:
            # Step 1: Initialize and validate all input parameters
            self._initialize_and_validate(float_model, representative_dataset, 
                                          framework, method, use_mixed_precision)

            # Step 2: Apply custom parameter modifications
            self._modify_params(param_items)

            # Step 3: Handle LQ-PTQ method separately (TensorFlow only)
            if self.method == 'LQPTQ':
                # Execute Low-bit Quantization Post-Training Quantization
                quantized_model = self._exec_lq_ptq()
                return True, quantized_model

            # Step 4: Select framework-specific quantization methods
            self._select_method()
            
            # Step 5: Select framework-specific argument names
            self._select_argname()

            # Step 6: Configure Target Platform Capabilities
            self._get_tpc()

            # Step 7: Prepare quantization parameters
            params_PTQ = self._setting_PTQparam()
            
            # Step 8: Execute quantization process (PTQ or GPTQ)
            quantized_model, _ = self._post_training_quantization(**params_PTQ)

            # Step 9: Export quantized model to specified format
            self._export_model(quantized_model)

            # Return success flag and quantized model
            return True, quantized_model

        except Exception as e:
            # Log error details and re-raise the exception to caller
            print(f"Error during quantization and export: {str(e)}")
            raise  # Re-raise the original exception to the caller

        finally:
            pass
