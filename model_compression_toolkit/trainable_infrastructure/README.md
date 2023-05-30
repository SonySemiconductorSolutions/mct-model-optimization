# Trainable Quantizers Infrastructure

The trainable infrastructure is a module containing quantization abstraction and quantizers for hardware-oriented model optimization tools such as the Model Compression Toolkit ([MCT](https://github.com/sony/model_optimization)).

It provides the required abstraction for trainable quantization methods such as quantization-aware training.

It utilizes the Inferable Quantizers Infrastructure provided by the [MCT Quantizers](https://github.com/sony/mct_quantizers) package, which proposes the required abstraction for emulating inference-time quantization.

## High level description

For each layer, we use a "Quantization Wrapper" to wrap the layer's weight quantizers, and an "Activation Quantization Holder" to hold the activation quantizers. 
Both components are provided by the [MCT Quantizers](https://github.com/sony/mct_quantizers) package.
We can choose the quantizers and all the quantization information for each layer by initializing the weights_quantizer and activation_quantizer API.

Notice that the quantization wrapper, holder and the quantizers are implemented per framework.

## Quantizers 
The quantizers in this module are built upon the "Inferable Quantizer" abstraction (from [MCT Quantizers](https://github.com/sony/mct_quantizers)), and define the "Trainable Quantizer" framework, which contains learnable quantization parameters that can be optimized during training.

## Details and Examples

More details and "how to" examples for TensorFlow can be found in:

[Trainable quantizers for TensorFlow](keras/README.md)

And for PyTorch:

[Trainable quantizers for PyTorch](pytorch/README.md)

  



