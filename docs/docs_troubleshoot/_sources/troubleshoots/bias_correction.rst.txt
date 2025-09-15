:orphan:

.. _ug-bias_correction:


================
Bias Correction
================

Overview
==============================
MCT applies bias correction by default to overcome induced bias shift caused by weights quantization. 

The applied correction is an estimation of the bias shift that is computed based on the collected statistical data generated with the representative dataset.

Therefore, the effect of the bias correction is sensitive to the distribution and size of the provided representative dataset.

Trouble Situation
==============================
The quantization accuracy may degrade when the representative dataset and bias correction are incompatible, causing a decrease in accuracy.

Solution
=================================
You can check if the bias correction causes a degradation in accuracy, by disabling the bias correction (setting ``weights_bias_correction`` to False of the ``QuantizationConfig`` in ``CoreConfig``).

.. code-block:: python

  core_config = mct.core.CoreConfig(mct.core.QuantizationConfig(weights_bias_correction=False))
  quantized_model, _ = mct.ptq.pytorch_post_training_quantization(...,
                                                                  core_config=core_config)

1. If you can increase your representative dataset size and its distribution, it may restore accuracy. 

2. If you don't have an option to increase or diversify your representative dataset, disabling the bias correction is recommended.