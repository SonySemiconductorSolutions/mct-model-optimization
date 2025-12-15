.. _ug-index:

========================================================================================
TroubleShooting Documentation (MCT XQuant Extension Tool)
========================================================================================


Overview
========

The Model Compression Toolkit (MCT) offers numerous functionalities to compress neural networks with minimal accuracy lost. However, in some cases, the compressed model may experience a significant decrease in accuracy. Fear not, as this lost accuracy can often be reclaimed by adjusting the quantization configuration or setup.

Outlined below are a series of steps aimed at recovering lost accuracy resulting from compression with MCT. Some steps may be applicable to your model, while others may not.


Quantization Troubleshooting for MCT[1]
============================================

**1. Judgeable Troubleshoots**

The following items are automatically identified by the XQuant Extension Tool.

Please read the following items indicated by XQuant Extension Tool, especially the **Solution** section.

* :ref:`Outlier Removal<ug-outlier_removal>`
* :ref:`Shift Negative Activation<ug-shift_negative_activation>`
* :ref:`Unbalanced "concatenation"<ug-unbalanced_concatenation>`
* :ref:`Mixed Precision with model output loss objective<ug-mixed_precision_with_model_output_loss_objective>`

**2. General Troubleshoots**

The following items are general troubleshoots for quantization accuracy improvement.

If quantization accuracy of your model does not improve after reading Judgeable Troubleshoots, please read the following items.

* :ref:`Representative Dataset size and  diversity<ug-representative_dataset_size_and_diversity>`
* :ref:`Representative and Validation Dataset Mismatch<ug-representative_and_validation_dataset_mismatch>`
* :ref:`Bias Correction<ug-bias_correction>`
* :ref:`Using more samples in Mixed Precision quantization<ug-using_more_samples_in_mixed_precision_quantization>`
* :ref:`Threshold selection error method<ug-threshold_selection_error_method>`
* :ref:`Enabling Hessian-based Mixed Precision<ug-enabling_hessian-based_mixed_precision>`
* :ref:`GPTQ - Gradient-Based Post Training Quantization<ug-gptq-gradient_based_post_training_quantization>`

.. note::
  
  In some pages, there are TensorBoard visualizations.
  You can make TensorBoard visualizations if you set ``mct.set_log_folder``. `Read more <https://sonysemiconductorsolutions.github.io/mct-model-optimization/guidelines/visualization.html>`_

References
============================================
[1] `Quantization Troubleshooting for MCT <https://github.com/SonySemiconductorSolutions/mct-model-optimization/tree/main/quantization_troubleshooting.md>`_

[2] `PyTorch documentation (v2.5) <https://docs.pytorch.org/docs/2.5/index.html>`_
