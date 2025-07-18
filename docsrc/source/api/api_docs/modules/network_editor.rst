:orphan:

.. _ug-network_editor:


=================================
network_editor Module
=================================

**The model can be edited by a list of EditRules to apply on nodes in a graph that represents the model during the model quantization. Each EditRule is a tuple of a filter and an action, where we apply the action on each node the filter matches**

EditRule
==========
.. autoclass:: model_compression_toolkit.core.network_editor.EditRule

Filters
==========

.. autoclass:: model_compression_toolkit.core.network_editor.NodeTypeFilter

|

.. autoclass:: model_compression_toolkit.core.network_editor.NodeNameFilter

|

.. autoclass:: model_compression_toolkit.core.network_editor.NodeNameScopeFilter


Actions
==========

.. autoclass:: model_compression_toolkit.core.network_editor.ChangeFinalWeightsQuantConfigAttr

|

.. autoclass:: model_compression_toolkit.core.network_editor.ChangeCandidatesWeightsQuantConfigAttr

|

.. autoclass:: model_compression_toolkit.core.network_editor.ChangeFinalActivationQuantConfigAttr

|

.. autoclass:: model_compression_toolkit.core.network_editor.ChangeCandidatesActivationQuantConfigAttr

|

.. autoclass:: model_compression_toolkit.core.network_editor.ChangeFinalWeightsQuantizationMethod

|

.. autoclass:: model_compression_toolkit.core.network_editor.ChangeCandidatesWeightsQuantizationMethod

|

.. autoclass:: model_compression_toolkit.core.network_editor.ChangeCandidatesActivationQuantizationMethod

