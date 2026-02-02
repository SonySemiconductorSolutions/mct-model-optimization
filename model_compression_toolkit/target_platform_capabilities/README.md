# Target Platform Capabilities (TPC)

## About 

TPC is our way of describing the hardware that will be used to run and infer with models that are
optimized using the MCT.

The TPC includes different parameters that are relevant to the
 hardware during inference (e.g., number of bits used
in some operator for its weights/activations, fusing patterns, etc.)


## Supported Target Platform Models 

Currently, MCT contains three target-platform models
(new models can be created and used by users as demonstrated [here](https://github.com/SonySemiconductorSolutions/mct-model-optimization/blob/main/model_compression_toolkit/target_platform_capabilities/tpc_models/imx500_tpc/v1_0/tpc.py)):
- [IMX500](https://developer.sony.com/develop/imx500/)
- [TFLite](https://www.tensorflow.org/lite/performance/quantization_spec)
- [QNNPACK](https://github.com/pytorch/QNNPACK)

The default target-platform model is [IMX500](https://developer.sony.com/develop/imx500/), quantizes activations using 8 bits with power-of-two thresholds for 
activations and symmetric threshold for weights.
For mixed-precision quantization it uses either 2, 4, or 8 bits for quantizing the operators.
One may view the full default target-platform model and its parameters [here](./tpc_models/imx500_tpc/v1_0/tpc.py).

[TFLite](./tpc_models/tflite_tpc/v1_0/tpc.py) and [QNNPACK](./tpc_models/qnnpack_tpc/v1_0/tpc.py) models were created similarly.

## Usage

The simplest way to initiate a TPC and use it in MCT is by using the function [get_target_platform_capabilities](https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/methods/get_target_platform_capabilities.html#ug-get-target-platform-capabilities).

This function gets a TPC object matching the tpc version and device type. Please check [here](https://github.com/SonySemiconductorSolutions/mct-model-optimization/blob/main/README.md#supported-versions) for supported versions.

For example:

```python
import model_compression_toolkit as mct

# Get the TPC object for imx500 hardware with version 1.0.
tpc = mct.get_target_platform_capabilities(tpc_version='1.0', device_type='imx500')

# Apply MCT on TensorFlow pre-trained model using the TPC.
quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(in_model=pretrained_model, # Replace with your pretrained model
                                                                              representative_data_gen=dataset, # Replace with your representative dataset
                                                                              target_platform_capabilities=tpc)
```

You can also get a TPC for IMX500 using the function [get_target_platform_capabilities_sdsp](https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/methods/get_target_platform_capabilities_sdsp.html#ug-get-target-platform-capabilities_sdsp) that specifies the sdsp converter version. Please check [here](https://github.com/SonySemiconductorSolutions/mct-model-optimization/blob/main/README.md#supported-versions) for supported versions.

For example:

```python
import model_compression_toolkit as mct

# Get the TPC object specified the sdsp converter version 3.14.
tpc = mct.get_target_platform_capabilities_sdsp(sdsp_version='3.14')

# Apply MCT on PyTorch pre-trained model using the TPC.
quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(in_module=pretrained_model, # Replace with your pretrained model
                                                                                representative_data_gen=dataset, # Replace with your representative dataset
                                                                                target_platform_capabilities=tpc)
```

For more information and examples, we highly recommend you to visit our [project website](https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/modules/target_platform_capabilities.html#ug-target-platform-capabilities).
