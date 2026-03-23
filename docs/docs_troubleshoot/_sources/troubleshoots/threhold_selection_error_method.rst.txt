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

  * NOCLIPPING          - Use min/max values as thresholds. This avoids clipping bias but reduces quantization resolution.
  * MSE (default)       - **(default)** Use mean square error for minimizing quantization noise.
  * MAE                 - Use mean absolute error for minimizing quantization noise.
  * KL                  - Use KL-divergence to make signals distributions to be similar as possible.
  * Lp                  - Use Lp-norm to minimizing quantization noise. The parameter p is specified by QuantizationConfig.l_p_value (default: 2; integer only). It equals MAE when p = 1 and MSE when p = 2. If you want to use p≧3, please use this method.
  * HMSE                - Use Hessian-based mean squared error for minimizing quantization noise. This method is using Hessian scores to factorize more valuable parameters when computing the error induced by quantization.

    **How to select QuantizationErrorMethod**

    .. csv-table::
        :header: "Method", "Recommended Situations"
        :widths: 20, 80

        "NOCLIPPING", "Research and debugging phases where you want to observe behavior across the entire range. This is effective when you want to maintain the entire range, especially when the data is biased (for example, when there is an extremely small amount of data on the minimum side)."
        "MSE", "**Basically, you should use this method.** This method is effective when the data distribution is close to normal and there are few outliers. Effective when you want stable results, such as in regression tasks."
        "MAE", "Effective for data with a lot of noise and outliers."
        "KL", "Useful for tasks where output distribution is important (such as Anomaly Detection)."
        "LP", "p≧3 is effective when you want to be more sensitive to outliers than MSE. (such as Sparse Data)."
        "HMSE", "Recommended when using GPTQ. This is effective for models where specific layers strongly influence the overall accuracy. (such as Transformers)."




For example, set NOCLIPPING to the ``activation_error_method`` attribute of the ``QuantizationConfig`` in ``CoreConfig``.

.. code-block:: python

    quant_config = mct.core.QuantizationConfig(activation_error_method=mct.core.QuantizationErrorMethod.NOCLIPPING)
    core_config = mct.core.CoreConfig(quantization_config=quant_config)
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(..., 
                                                                    core_config=core_config)

.. note::

    Some error methods (specifically, the KL-divergence method) may suffer from extended runtime periods. 
    Opting for a different error metric could enhance threshold selection for one layer while potentially compromising another. 
