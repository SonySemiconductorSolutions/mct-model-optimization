:orphan:

.. _ug-using_more_samples_in_mixed_precision_quantization:


===================================================
Using more samples in Mixed Precision quantization
===================================================
In Mixed Precision quantization, MCT will assign a different bit width to each weight in the model, depending on the weight's layer sensitivity and a resource constraint defined by the user, such as target model size.

Check out the `mixed precision tutorial <https://github.com/SonySemiconductorSolutions/mct-model-optimization/tree/main/tutorials/notebooks/mct_features_notebooks/pytorch/example_pytorch_mixed_precision_ptq.ipynb>`_ for more information.

Overview
==============================
By default, MCT employs 32 samples from the provided representative dataset for the Mixed Precision search. Leveraging a larger dataset could enhance results, particularly when dealing with datasets exhibiting high variance.

Trouble Situation
==============================
The quantization accuracy may degrade when using Mixed Precision quantization with a small number of samples.

Solution
=================================
Increase the number of samples (e.g. to 64 samples).

Set the ``num_of_images`` attribute to a larger value of the ``MixedPrecisionQuantizationConfig`` in ``CoreConfig``.

.. code-block:: python

  mixed_precision_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=64)
  core_config = mct.core.CoreConfig(mixed_precision_config=mixed_precision_config)
  quantized_model, _ = mct.ptq.pytorch_post_training_quantization(..., 
                                                                  core_config=core_config)

.. note::

  Expanding the sample size may lead to extended runtime during the Mixed Precision search process.
