# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, List, Dict, TYPE_CHECKING
from enum import Enum, auto

from model_compression_toolkit.core.common.framework_info import ChannelAxisMapping
from model_compression_toolkit.logger import Logger

from model_compression_toolkit.target_platform_capabilities.constants import POSITIONAL_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import \
    AttributeQuantizationConfig, OpQuantizationConfig

if TYPE_CHECKING:
    from model_compression_toolkit.core.common.graph.base_node import WeightAttrT

##########################################
# Every node holds a quantization configuration
# for its weights and activations quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################


class ActivationQuantizationMode(Enum):
    """ An enum defining the output activation quantization mode of  a node. """
    QUANT = auto()
    FLN_QUANT = auto()
    PRESERVE_QUANT = auto()
    NO_QUANT = auto()
    FLN_NO_QUANT = auto()


class BaseNodeQuantizationConfig(object):
    """
    Base class for node quantization configuration
    """

    def set_quant_config_attr(self, config_parameter_name: str, config_parameter_value: Any,
                              *args: List[Any], **kwargs: Dict[str, Any]):
        """
        Changes a BaseNodeQuantizationConfig's parameter.
        Note that arg and kwargs are only to allow clean override in the child classes.

        Args:
            config_parameter_name: parameter name to change.
            config_parameter_value: parameter value to change.
            args: A list of additional arguments.
            kwargs: A dictionary with additional key arguments.

        """
        if hasattr(self, config_parameter_name):
            setattr(self, config_parameter_name, config_parameter_value)
        else:
            raise AttributeError(
                f"Parameter {config_parameter_name} could not be found in the node quantization config.")

    def __repr__(self) -> str:
        """
        Returns: String to display a NodeQuantizationConfig object.
        """
        # Used for debugging, thus no cover.
        return ''.join(f'{k}: {v}\n' for k, v in self.__dict__.items())  # pragma: no cover


class NodeActivationQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Attributes for configuring the quantization of the activations of a node.
    """
    def __init__(self, op_cfg: OpQuantizationConfig):
        """

        Args:
            op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
        """
        self.activation_quantization_method = op_cfg.activation_quantization_method
        self.activation_n_bits = op_cfg.activation_n_bits
        if op_cfg.enable_activation_quantization and op_cfg.quantization_preserving:
            raise ValueError("An OpQuantizationConfig can't have both enable_activation_quantization and quantization_preserving enabled.")
        if op_cfg.enable_activation_quantization:
            self.quant_mode = ActivationQuantizationMode.QUANT
        elif op_cfg.quantization_preserving:
            self.quant_mode = ActivationQuantizationMode.PRESERVE_QUANT
        else:
            self.quant_mode = ActivationQuantizationMode.NO_QUANT
        self.signedness = op_cfg.signedness

        self.activation_quantization_params = {}
        # TODO: computed by compute_activation_bias_correction. Probably shouldnt be here.
        self.activation_bias_correction_term = None
        # Z-threshold is a global param from QuantizationConfig, however it can be overridden per node by NetworkEditor.
        # Since activation qparams are re-computed in several places, it's easier to keep it here and update it once.
        self.z_threshold = None

    @property
    def enable_activation_quantization(self):
        return self.quant_mode == ActivationQuantizationMode.QUANT

    @property
    def quantization_preserving(self):
        return self.quant_mode == ActivationQuantizationMode.PRESERVE_QUANT

    def fln_quantization(self):
        return self.quant_mode == ActivationQuantizationMode.FLN_QUANT

    def set_activation_quantization_param(self,
                                          activation_params: dict):
        """
         Set a quantization parameter for the node's activation.

        Args:
            activation_params: Dictionary that contains weight quantization params.

        """
        assert self.quant_mode == ActivationQuantizationMode.QUANT or self.quant_mode == ActivationQuantizationMode.FLN_QUANT
        for param_name, param_value in activation_params.items():
            self.activation_quantization_params[param_name] = param_value

    def __eq__(self, other: Any) -> bool:
        """
        Compares the object to another object to find if they are equal.

        Args:
            other: An object to compare to.

        Returns: Whether the objects are identical or not.

        """
        if not isinstance(other, NodeActivationQuantizationConfig):
            return False  # pragma: no cover

        return self.activation_quantization_method == other.activation_quantization_method and \
               self.activation_n_bits == other.activation_n_bits and \
               self.quant_mode == other.quant_mode and \
               self.signedness == other.signedness

    def __hash__(self):
        return hash((self.activation_quantization_method,
                     self.activation_n_bits,
                     self.quant_mode,
                     self.signedness))


class WeightsAttrQuantizationConfig:
    """
    Configuration for quantizing a weights attribute of a node.
    """
    def __init__(self,
                 weights_attr_cfg: AttributeQuantizationConfig,
                 weights_channels_axis: ChannelAxisMapping = None):
        """

        Args:
            weights_attr_cfg: AttributeQuantizationConfig with parameters to use when creating the node's attribute quantization config.
            weights_channels_axis: Axis to quantize a node's attribute when quantizing per-channel (if not quantizing per-channel than expecting None).
        """
        self.weights_channels_axis = weights_channels_axis
        self.weights_quantization_method = weights_attr_cfg.weights_quantization_method
        self.weights_n_bits = weights_attr_cfg.weights_n_bits
        self.weights_per_channel_threshold = weights_attr_cfg.weights_per_channel_threshold
        self.enable_weights_quantization = weights_attr_cfg.enable_weights_quantization

        self.weights_quantization_params = {}

    def set_weights_quantization_param(self,
                                       weights_params: dict):
        """
         Set a quantization parameter for the node's weights.

        Args:
            weights_params: Dictionary that contains weight quantization params.

        """
        assert self.enable_weights_quantization
        for param_name, param_value in weights_params.items():
            self.weights_quantization_params[param_name] = param_value

    def __eq__(self, other: Any) -> bool:
        """
        Compares the object to another object to find if they are equal.

        Args:
            other: An object to compare to.

        Returns: Whether the objects are identical or not.

        """
        if not isinstance(other, WeightsAttrQuantizationConfig):
            return False  # pragma: no cover

        return self.weights_channels_axis == other.weights_channels_axis and \
               self.weights_quantization_method == other.weights_quantization_method and \
               self.weights_n_bits == other.weights_n_bits and \
               self.weights_per_channel_threshold == other.weights_per_channel_threshold and \
               self.enable_weights_quantization == other.enable_weights_quantization

    def __hash__(self):
        return hash((self.weights_channels_axis,
                     self.weights_quantization_method,
                     self.weights_n_bits,
                     self.weights_per_channel_threshold,
                     self.enable_weights_quantization))


class NodeWeightsQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Holding a mapping between the node's weights attributes and their quantization configurations,
    in addition to quantization parameters that are global for all attributes of the represented node.
    """
    def __init__(self,
                 op_cfg: OpQuantizationConfig,
                 weights_channels_axis: ChannelAxisMapping,
                 node_attrs_list: List[str]):
        """

        Args:
            op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
            weights_channels_axis: Axis to quantize a node's weights attribute when quantizing per-channel.
            node_attrs_list: A list of the node's weights attributes names.

        """
        self.simd_size = op_cfg.simd_size

        # Initialize a quantization configuration for each of the node's attributes
        self.attributes_config_mapping = {}
        self.pos_attributes_config_mapping = {}
        for attr in node_attrs_list:
            if isinstance(attr, int):
                # this is a positional attribute, so it needs to be handled separately.
                # Search for any keys in the op config's attribute weight config mapping that contain the
                # POS_ATTR string. If none are found, it indicates that no specific quantization config is defined for
                # positional weights, so the default config will be used instead.
                attrs_included_in_name = {k: v for k, v in op_cfg.attr_weights_configs_mapping.items() if
                                          POSITIONAL_ATTR in k}

                if len(attrs_included_in_name) > 1:  # pragma: no cover
                    raise ValueError(f"Found multiple attribute in FQC OpConfig that are contained "
                                     f"in the attribute name '{attr}'."
                                     f"Please fix the FQC attribute names mapping such that each operator's attribute"
                                     f" would have a unique matching name.")

                # If no specific positional attribute config is found, fall back to the default weight attribute config.
                if len(attrs_included_in_name) == 0:
                    attr_cfg = op_cfg.default_weight_attr_config
                else:
                    # If a specific config was found using POS_ATTR, use it.
                    attr_cfg = list(attrs_included_in_name.values())[0]

                # Register this attribute under the positional attributes config mapping.
                self.pos_attributes_config_mapping[attr] = WeightsAttrQuantizationConfig(weights_attr_cfg=attr_cfg,
                                                                                         weights_channels_axis=
                                                                                         weights_channels_axis)
            else:
                # In Tensorflow, the attribute name is composed of the framework attribute name and the layer name,
                # therefore, we need to look for the attribute in the op_cfg that is contained in the node attribute's name.
                attrs_included_in_name = {k: v for k, v in op_cfg.attr_weights_configs_mapping.items() if k in attr}
                if len(attrs_included_in_name) > 1:  # pragma: no cover
                    Logger.critical(f"Found multiple attribute in FQC OpConfig that are contained "
                                    f"in the attribute name '{attr}'."
                                    f"Please fix the FQC attribute names mapping such that each operator's attribute would "
                                    f"have a unique matching name.")
                if len(attrs_included_in_name) == 0:
                    attr_cfg = op_cfg.default_weight_attr_config
                else:
                    attr_cfg = list(attrs_included_in_name.values())[0]

                self.attributes_config_mapping[attr] = WeightsAttrQuantizationConfig(weights_attr_cfg=attr_cfg,
                                                                                     weights_channels_axis=weights_channels_axis)
        # TODO this is set by batch norm reconstruction substitution when folded batch norms are added back, to mark
        #  the nodes that the correction should be applied to (for some nodes it gets disabled) and BNs removed.
        #  The actual correction is only computed when it's applied in ptq, so it seems that both substitutions could
        #  be unified, and no info need to pass between.
        self.weights_second_moment_correction = None
        # TODO: computed corrected bias is injected to the node config. Probably shouldn't be here. Also it can be
        #  computed on the final config, instead of all candidates and then there is no need to save it at all.
        self.bias_corrected = None

    def get_attr_config(self, attr_name: 'WeightAttrT') -> WeightsAttrQuantizationConfig:
        """
        Returns a weights attribute config for an attribute that contains the given name.
        If multiple attributes that contain the given name are found - looking for the exact name, otherwise,
        fails with an error message.
        If none attributes that contain the given name are found - fails with an error message.

        Args:
            attr_name: The name of the attribute to get its quantization configuration.

        Returns: An attribute quantization configuration.

        """
        if attr_name is None:  # pragma: no cover
            Logger.critical("Got 'None' attribute name for retrieving weights attribute quantization configuration.")

        if isinstance(attr_name, int):
            # this is a positional attribute
            attr_cfg = self.pos_attributes_config_mapping.get(attr_name)
        else:
            attrs_with_name = self._extract_config_for_attributes_with_name(attr_name)
            attr_cfg = None
            if len(attrs_with_name) == 1:
                attr_cfg = [v for v in attrs_with_name.values()][0]
            elif len(attrs_with_name) > 1:
                Logger.warning(f"Found multiple weight attributes containing the name {attr_name}: "
                               f"{list(attrs_with_name.keys())}. Looking for an attributes with the exact name.")
                # If no attribute with the exact name then an error would be thrown
                attr_cfg = self.attributes_config_mapping.get(attr_name)

        if attr_cfg is None:  # pragma: no cover
            Logger.critical(f"Weight attribute '{attr_name}' config could not be found.")

        return attr_cfg

    def set_attr_config(self, attr_name: 'WeightAttrT', attr_qc: WeightsAttrQuantizationConfig):
        """
        Adding a new attribute with quantization configuration to the node's weights configurations mapping.

        Args:
            attr_name: The name of the attribute to set a quantization configuration to.
            attr_qc: The quantization configuration to set.

        """
        if isinstance(attr_name, int):
            self.pos_attributes_config_mapping[attr_name] = attr_qc
        else:
            self.attributes_config_mapping[attr_name] = attr_qc

    def has_attribute_config(self, attr_name: 'WeightAttrT') -> bool:
        """
        Checks whether the node weights configuration contains a configuration for a given weights attribute.

        Args:
            attr_name: The attribute name to check if a quantization configuration is defined for.

        Returns: True if the attribute exists in the attributes configuration mapping, False otherwise.

        """
        if isinstance(attr_name, int):
            return self.pos_attributes_config_mapping.get(attr_name, False)
        else:
            saved_attr_name = self._extract_config_for_attributes_with_name(attr_name)
            if len(saved_attr_name) >= 1:
                return True

        return False

    @property
    def all_weight_attrs(self) -> List['WeightAttrT']:
        """ Fetch all weight attributes keys (positional and named).

            Returns:
                List of attributes.
        """
        return list(self.pos_attributes_config_mapping.keys()) + list(self.attributes_config_mapping.keys())

    def get_all_weight_attrs_configs(self) -> Dict['WeightAttrT', AttributeQuantizationConfig]:
        """ Get quantization configs for all weights.

            Returns:
                A dict from weight attribute to its config.
        """
        return {attr: self.get_attr_config(attr) for attr in self.all_weight_attrs}

    def disable_all_weights_quantization(self):
        """ Disable quantization for all weights. """
        for w_cfg in self.pos_attributes_config_mapping.values():
            w_cfg.enable_weights_quantization = False
        for w_cfg in self.attributes_config_mapping.values():
            w_cfg.enable_weights_quantization = False

    def _extract_config_for_attributes_with_name(self, attr_name) -> Dict[str, WeightsAttrQuantizationConfig]:
        """
        Extract the saved attributes that contain the given attribute name.
        Relevant to Tensorflow where attributes are presented with the layer name and index,
        in addition to the attribute actual name.

        Args:
            attr_name: An attribute to extract its saved name.

        Returns: A mapping between attributes that contain the given name to their configuration.

        """
        attrs_with_name = {k: v for k, v in self.attributes_config_mapping.items() if attr_name in k}
        if len(attrs_with_name) > 1:
            Logger.warning(f"Found multiple weight attributes containing the name {attr_name}: "
                           f"{list(attrs_with_name.keys())}.")
        return attrs_with_name

    def set_quant_config_attr(self, config_parameter_name: str, config_parameter_value: Any,
                              attr_name: 'WeightAttrT' = None, *args: List[Any], **kwargs: Dict[str, Any]):
        """
        This method overrides the parent class set_quant_config_attr to enable setting a specific weights
        attribute config parameter.

        Args:
            attr_name: attribute name to change.
            config_parameter_name: parameter name to change.
            config_parameter_value: parameter value to change.
            args: A list of additional arguments.
            kwargs: A dictionary with additional key arguments.

        """

        if attr_name is None:
            super(NodeWeightsQuantizationConfig, self).set_quant_config_attr(config_parameter_name,
                                                                             config_parameter_value,
                                                                             *args, **kwargs)
        else:
            if self.has_attribute_config(attr_name):
                attr_cfg = self.get_attr_config(attr_name)
                if hasattr(attr_cfg, config_parameter_name):
                    setattr(attr_cfg, config_parameter_name, config_parameter_value)
                else:
                    raise AttributeError(f"Parameter {config_parameter_name} could not be found in the node quantization config of "
                                         f"weights attribute {attr_name}.")
            else:  # pragma: no cover
                Logger.critical(f"Weights attribute {attr_name} could not be found to set parameter {config_parameter_name}.")

    def __eq__(self, other: Any) -> bool:
        """
        Compares the object to another object to find if they are equal.

        Args:
            other: An object to compare to.

        Returns: Whether the objects are identical or not.

        """
        if not isinstance(other, NodeWeightsQuantizationConfig):
            return False  # pragma: no cover

        return self.simd_size == other.simd_size and \
            self.attributes_config_mapping.keys() == other.attributes_config_mapping.keys() and \
            all([self.attributes_config_mapping[k] == other.attributes_config_mapping[k]
                 for k in self.attributes_config_mapping.keys()]) and \
            self.pos_attributes_config_mapping.keys() == other.pos_attributes_config_mapping.keys() and \
            all([self.pos_attributes_config_mapping[k] == other.pos_attributes_config_mapping[k]
                 for k in self.pos_attributes_config_mapping.keys()])

    def __hash__(self):
        return hash((self.simd_size,
                     frozenset(self.attributes_config_mapping),
                     frozenset(self.pos_attributes_config_mapping)))
