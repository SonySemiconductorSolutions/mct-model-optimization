# Task notebooks
This set of tutorials provide for various common deep learning applications(Object detection, Semantic Segmentation, etc).
The notebooks in this section demonstrate how to use MCT for various tasks and models.

#### Python Version Downgrade for Google Colab (as of December 2025)

The default Python version in Google Colab (3.12 or later) does not support TensorFlow 2.15 or earlier.
If you want to run tutorials that use them, please downgrade to Python 3.11 by following these steps:

1. In your Colab notebook, select **Runtime** → **Change runtime type** from the menu

2. In the "Change runtime type" dialog, set **Runtime version** to `2025.07`

### Keras Tutorials

  | Model                                                                      | Task                      | Notes                                                                                        |
  |----------------------------------------------------------------------------|---------------------------|----------------------------------------------------------------------------------------------|
  | [EfficientDet](keras/example_effdet_keras_mixed_precision_ptq.ipynb)       | Object Detection          | use [CustomLayer](https://github.com/SonySemiconductorSolutions/aitrios-edge-mdt-cl/tree/main)    |

### Pytorch Tutorials

  | Model                                                                      | Task                      | Notes                                                                                             |
  |----------------------------------------------------------------------------|---------------------------|---------------------------------------------------------------------------------------------------|
  | [PoseNet](pytorch/example_posenet_pytorch_mixed_precision_ptq.ipynb)       | Human Pose Estimation     |                                                                                                   |
  | [YOLOX](pytorch/example_yolox_pytorch_mixed_precision_ptq.ipynb)           | Object Detection          | use [CustomLayer](https://github.com/SonySemiconductorSolutions/aitrios-edge-mdt-cl/tree/main)    |
  | [DeepLabv3+](pytorch/example_deeplabv3p_pytorch_mixed_precision_ptq.ipynb) | Semantic Segmentation     |                                                                                                   |
