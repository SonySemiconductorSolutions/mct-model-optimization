:orphan:

.. _ug-gptq-gradient_based_post_training_quantization:


==================================================
GPTQ - Gradient-Based Post Training Quantization
==================================================

Overview
==============================
When PTQ (either with or without Mixed Precision) fails to deliver the required accuracy, GPTQ is potentially the remedy. 

In GPTQ, MCT will finetune the model's weights and quantization parameters for improved accuracy. The finetuning process will only use the label-less representative dataset.

Check out the `GPTQ tutorial <https://github.com/SonySemiconductorSolutions/mct-model-optimization/tree/main/tutorials/notebooks/mct_features_notebooks/pytorch/example_pytorch_mobilenet_gptq.ipynb>`_ for more information and an implementation example.

Solution
=================================
MCT can configure GPTQ optimization options, such as the number of epochs for the optimization process.

For example, set the number of epochs to 50.

.. code-block:: python

  gptq_config = mct.gptq.get_pytorch_gptq_config(n_epochs=50)
  quantized_model, _ = mct.gptq.pytorch_gradient_post_training_quantization(model, representative_dataset,
                                                                            gptq_config=gptq_config)

.. note::

  * The finetuning process will take much longer to finish than PTQ. As in any finetuning, some hyperparameters optimization may be required.

  * You can use Mixed Precision and GPTQ together.
