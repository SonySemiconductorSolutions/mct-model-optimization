:orphan:

.. _ug-enabling_hessian-based_mixed_precision:


============================================
Enabling Hessian-based Mixed Precision
============================================
In Mixed Precision quantization, MCT will assign a different bit width to each weight in the model, depending on the weight's layer sensitivity and a resource constraint defined by the user, such as target model size.

Check out the `Mixed Precision tutorial <https://github.com/SonySemiconductorSolutions/mct-model-optimization/blob/v2.4.2/tutorials/notebooks/mct_features_notebooks/pytorch/example_pytorch_mixed_precision_ptq.ipynb>`_ for more information.

Overview
==============================
MCT offers a Hessian-based scoring mechanism to assess the importance of layers during the Mixed Precision search. 

This feature can notably enhance Mixed Precision outcomes for certain network architectures.

Solution
=================================
Set the ``use_hessian_based_scores`` flag to True in the ``MixedPrecisionQuantizationConfig`` of the ``CoreConfig``.

.. code-block:: python

    mixed_precision_config = mct.core.MixedPrecisionQuantizationConfig(use_hessian_based_scores=True)
    core_config = mct.core.CoreConfig(mixed_precision_config=mixed_precision_config)
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(..., 
                                                                    core_config=core_config)

.. note::

    Computing Hessian scores can be computationally intensive, potentially leading to extended Mixed Precision search runtime.
    Furthermore, these scoring methods may introduce unexpected noise into the Mixed Precision process, necessitating a deeper understanding of the underlying mechanisms and potential recalibration of program parameters.
