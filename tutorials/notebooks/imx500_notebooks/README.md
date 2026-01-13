# IMX500 notebooks
This set of tutorials provide how to quantize pre-trained models, especiallay for IMX500.

#### Python Version Downgrade for Google Colab (as of December 2025)

The default Python version in Google Colab (3.12 or later) does not support TensorFlow 2.15 or earlier.
If you want to run tutorials that use them, please downgrade to Python 3.11 by following these steps:

1. In your Colab notebook, select **Runtime** → **Change runtime type** from the menu

2. In the "Change runtime type" dialog, set **Runtime version** to `2025.07`

### Keras Tutorials

  | Tutorial                                                                   |
  |----------------------------------------------------------------------------|
  | [Basic Post-Training Quantization (PTQ) and Exporter](keras/example_keras_mobilenetv2_for_imx500.ipynb)       |
