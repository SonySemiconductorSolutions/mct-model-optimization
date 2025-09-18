:orphan:

.. _ug-representative_dataset_size_and_diversity:


============================================
Representative Dataset size and diversity
============================================

Overview
==============================
The representative dataset is used by the MCT to derive the threshold values of activation tensors in the model.

.. code-block:: python

    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(model, representative_dataset)

1. If the representative dataset size is too small, the thresholds will overfit and the accuracy on the validation dataset will degrade.
2. A similar overfitting may occur when the representative dataset isn't diverse enough (e.g. images from a single class, in a classification model). 
   In this case, the distribution of the target dataset and the representative dataset will not match, which might cause an accuracy degradation.

Trouble Situation
==============================
The quantization accuracy may degrade when:

1. The representative dataset size is too small.
2. The representative dataset has diversity (e.g. enough images for each class, in a classification model).

Solution
=================================
1. Increase the number of samples in the representative dataset.
2. Make sure that the samples are more diverse (e.g. include samples from all the classes, in a classification model).
