:orphan:

.. _ug-XQuantConfig:

================================================
XQuant Configuration
================================================

.. autoclass:: model_compression_toolkit.xquant.common.xquant_config.XQuantConfig
    :members:

.. note::

    The following parameters are only used in the **xquant_report_troubleshoot_pytorch_experimental** function.

    - quantize_reported_dir
    - threshold_quantize_error 
    - is_detect_under_threshold_quantize_error
    - threshold_degrade_layer_ratio
    - threshold_zscore_outlier_removal
    - threshold_ratio_unbalanced_concatenation
    - threshold_bitwidth_mixed_precision_with_model_output_loss_objective