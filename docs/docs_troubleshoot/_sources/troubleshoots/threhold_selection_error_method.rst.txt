:orphan:

.. _ug-threshold_selection_error_method:


=================================
Threshold selection error method
=================================

Overview
==============================
The quantization threshold, which determines how data gets quantized, involves an optimization process driven by predefined objective metrics. 

MCT defaults to employing the Mean-Squared Error (MSE) metric for threshold optimization, 

however, it offers a range of alternative error metrics (e.g. using min/max values, KL-divergence, etc.) to accommodate different network requirements.

This flexibility becomes particularly crucial for activation quantization, where threshold selection spans the entire tensor and relies on statistical insights for optimization. 

We advise you to consider other error metrics if your model is suffering from significant accuracy degradation, especially if it contains unorthodox activation layers.

Solution
=================================
Use a different error method for activations.  You can set the following values:

  * NOCLIPPING          - Use min/max values
  * MSE (default)       - Use mean square error 
  * MAE                 - Use mean absolute error
  * KL                  - Use KL-divergence
  * Lp                  - Use Lp-norm
  * HMSE                - Use Hessian-based mean squared error

For example, set NOCLIPPING to the ``activation_error_method`` attribute of the ``QuantizationConfig`` in ``CoreConfig``.

.. code-block:: python

    quant_config = mct.core.QuantizationConfig(activation_error_method=mct.core.QuantizationErrorMethod.NOCLIPPING)
    core_config = mct.core.CoreConfig(quantization_config=quant_config)
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(..., 
                                                                    core_config=core_config)

.. note::

    Some error methods (specifically, the KL-divergence method) may suffer from extended runtime periods. 
    Opting for a different error metric could enhance threshold selection for one layer while potentially compromising another. 
