:orphan:

.. _ug-Representative_and_Validation_Dataset_Mismatch:


==============================================
Representative and Validation Dataset Mismatch
==============================================

Overview
==============================
The representative dataset is used by the MCT to derive the threshold values of activation tensors in the model.

.. code-block:: python

    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(model, representative_dataset)

Usually, the representative dataset is taken from the training set, and uses the same preprocessing as the validation set. 

If that's not the case, accuracy degradation is expected.

Trouble Situation
==============================
The Quantization accuracy may degrade when the preprocessing of the representative dataset is not identical to the validation dataset, or its images are taken from the different domain.

Solution
=================================
The representative and validation datasets should come from the same domain and have the same preprocessing applied to them.
