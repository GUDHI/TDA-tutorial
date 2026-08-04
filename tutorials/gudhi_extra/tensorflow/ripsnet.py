# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
# See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
# Author(s):    Felix Hensel, Mathieu Carrière
#
# Copyright (C) 2022 Inria
#
# Modification(s):
#   - YYYY/MM Author: Description of the modification

import tensorflow as tf

class DenseRagged(tf.keras.layers.Layer):
    """
    This is a class for the ragged layer in the RipsNet architecture, processing the input pointclouds.
    """

    def __init__(self, units, input_dim=None, use_bias=True, activation='gelu', kernel_initializer=None,
                 bias_initializer=None, **kwargs):
        """
        Constructor for the DenseRagged class.

        Parameters:
            units (int): number of units in the layer.
            use_bias (bool): flag, indicating whether to use bias or not.
            activation (string or function): identifier of a keras activation function, e.g. 'relu'.
            kernel_initializer: tensorflow kernel initializer.
            bias_initializer: tensorflow bias initializer.
        """
        super().__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'units': self.units,
                'use_bias': self.use_bias,
                'activation': self.activation,
                '_supports_ragged_inputs': self._supports_ragged_inputs,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
            }
        )
        return config

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel', shape=[last_dim, self.units], trainable=True,
                                      initializer=self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=[self.units, ], trainable=True,
                                        initializer=self.bias_initializer)
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        """
        Apply DenseRagged layer on a ragged input tensor.

        Parameters:
            ragged tensor (e.g. containing a point cloud).

        Returns:
            ragged tensor containing the output of the layer.
        """
        outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
        outputs = tf.ragged.map_flat_values(self.activation, outputs)
        return outputs


class TFBlock(tf.keras.layers.Layer):
    """
    This class is a block of tensorflow layers.
    If the first layer is an instance of DenseRagged, it will automatically support ragged inputs.

    Parameters:
    layers (list): a list of either tensorflow layers or DenseRagged layers :class:`~gudhi.tensorflow.DenseRagged`.
    input_dim (int): dimension of the point cloud, if the input consists of point clouds.
    """

    def __init__(self, layers, **kwargs):
        """
        Constructor for the TFBlock class.

        Parameters:
        dense_layers (list): a list of dense tensorflow layers.
        """
        super().__init__(**kwargs)
        self.layers = layers
        if isinstance(layers[0], DenseRagged):
            self._supports_ragged_inputs = True

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'layers': self.layers,
                '_supports_ragged_inputs': self._supports_ragged_inputs,
            }
        )
        return config

    def build(self, input_shape):
        return self

    def call(self, inputs):
        """
        Apply the sequence of layers on an input tensor.

        Parameters:
        inputs: any input tensor.

        Returns:
        output tensor containing the output of the sequence of layers.
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


class PermopRagged(tf.keras.layers.Layer):
    """
    This is a class for the permutation invariant layer in the RipsNet architecture.
    """

    def __init__(self, perm_op, **kwargs):
        """
        Constructor for the PermopRagged class.

        Parameters:
            perm_op: permutation invariant function, such as `tf.math.reduce_sum`, `tf.math.reduce_mean`.
        """
        super().__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.perm_op = perm_op

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'perm_op': self.perm_op,
                '_supports_ragged_inputs': self._supports_ragged_inputs,
            }
        )
        return config

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        """
        Apply PermopRagged on an input tensor.
        """
        out = self.perm_op(inputs, axis=1)
        return out


class RipsNet(tf.keras.Model):
    """
    This is a TensorFlow model for estimating vectorizations of persistence diagrams of point clouds.
    This class implements the RipsNet described in the following article <https://arxiv.org/abs/2202.01725>.
    """

    def __init__(self, phi_1, phi_2, input_dim, perm_op='mean', **kwargs):
        """
        Constructor for the RipsNet class.

        Parameters:
            phi_1 (layers): any block of DenseRagged layers. Can be a custom block built from
                :class:`~gudhi.tensorflow.DenseRagged` layers.
            phi_2 (layers): Can be any (block of) TensorFlow layer(s), e.g. :class:`~gudhi.tensorflow.TFBlock`.
            input_dim (int): dimension of the input point clouds.
            perm_op (str or function): Permutation invariant operation.
                                       Can be 'mean' or 'sum', or any user defined (permutation invariant) function.
        """
        super().__init__(**kwargs)
        self.phi_1 = phi_1
        self.perm_op = perm_op
        self.phi_2 = phi_2
        self.input_dim = input_dim

        if perm_op not in ['mean', 'sum']:
            raise ValueError(f'Permutation invariant operation: {self.perm_op} is not allowed, must be "mean" or "sum".')

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'phi_1': self.phi_1,
                'phi_2': self.phi_2,
                'perm_op': self.perm_op,
                'input_dim': self.input_dim,
            }
        )
        return config

    def build(self, input_shape):
        return self

    def call(self, pointclouds):
        """
        Apply RipsNet on a ragged tensor containing a list of pointclouds.

        Parameters:
            pointclouds (n x None x input_dimension): ragged tensor containing n pointclouds in dimension
                `input_dimension`. The second dimension is ragged since point clouds can have different numbers of
                points.

        Returns:
            output (n x output_shape): tensor containing predicted vectorizations of the persistence diagrams of
                pointclouds.
        """
        if self.perm_op == 'mean':
            perm_op_ragged = PermopRagged(tf.math.reduce_mean)
        elif self.perm_op == 'sum':
            perm_op_ragged = PermopRagged(tf.math.reduce_sum)

        inputs = tf.cast(pointclouds, dtype="float32")
        output = self.phi_1(inputs)
        output = perm_op_ragged(output)
        output = self.phi_2(output)
        return output
