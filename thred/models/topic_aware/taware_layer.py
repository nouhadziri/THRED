import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs


class MultiDenseLayer(layers_base.Layer):
    def __init__(self, units, activation=None, use_bias=True, dtype=None, name=None, scope=None, **kwargs):
        super(MultiDenseLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self.units = units

        self.activation = activation
        self.use_bias = use_bias
        self._scope = scope

        self._built = False

        self._n_tensors = None
        self.kernels = None
        self.bias = None

    def build(self, input_shapes):
        if self._built:
            return

        self.kernels = []
        self.bias = None
        self._n_tensors = len(input_shapes)

        with vs.variable_scope(self._scope or self._name) as scope:
            with ops.name_scope(scope.original_name_scope):
                for i, input_shape in enumerate(input_shapes):
                    self.kernels.append(vs.get_variable('kernel_{}'.format(i),
                                                        shape=[input_shape[-1], self.units],
                                                        dtype=self._dtype,
                                                        trainable=True))

                if self.use_bias:
                    self.bias = vs.get_variable('bias',
                                           shape=[self.units, ],
                                           initializer=init_ops.zeros_initializer(),
                                           dtype=self._dtype, trainable=True)

        self._built = True

    def __op(self, kernel, inputs, shape):
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, kernel, [[len(shape) - 1],[0]])
            # Reshape the output back to the original ndim of the input.
            # if context.in_graph_mode():
            # for tf > 1.5.0
            if not context.executing_eagerly():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = tf.matmul(inputs, kernel)

        return outputs

    def __call__(self, inputs, *args, **kwargs):
        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            raise ValueError('input must be a list')

        if not inputs:
            raise ValueError('input cannot be empty')

        input_shapes = [inp.get_shape().as_list() for inp in inputs]

        self.build(input_shapes)

        outputs = None
        for i, kernel in enumerate(self.kernels):
            out = self.__op(kernel, inputs[i], input_shapes[i])
            if outputs is None:
                outputs = out
            else:
                outputs = tf.add(out, outputs)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class JointDenseLayer(layers_base.Layer):
    def __init__(self, vocab_size, topic_vocab_size, dtype=None, name=None, scope=None, **kwargs):
        super(JointDenseLayer, self).__init__(name=name, dtype=dtype, **kwargs)
        self._vocab_size = vocab_size
        self._topic_vocab_size = topic_vocab_size
        self._units = vocab_size

        with tf.variable_scope(scope or "output_projection"):
            self._msg_layer = MultiDenseLayer(
                self._vocab_size,
                # activation=tf.nn.tanh,
                name="message_projection")
            self._topical_layer = MultiDenseLayer(
                self._topic_vocab_size,
                # activation=tf.nn.tanh,
                name="topical_projection")

        self._scope = scope

        self._built = False

        self._n_tensors = None
        self.kernels = None
        self.bias = None

    def compute_output_shape(self, input_shape):
        # remove "_" from the name for tf > 1.5.0
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self._units)

    def __call__(self, rnn_output, *args, **kwargs):
        input = kwargs.get('input')
        context_attention = kwargs.get('context')

        message_outputs = self._msg_layer([rnn_output, input])

        shape = rnn_output.get_shape().as_list()
        pad_dims = [[0] * 2 for _ in range(len(shape))]
        pad_dims[-1][0] = self._vocab_size - self._topic_vocab_size

        if context_attention is None:
            topical_layer_inputs = [rnn_output, input]
        else:
            topical_layer_inputs = [rnn_output, input, context_attention]

        topical_outputs = tf.pad(self._topical_layer(topical_layer_inputs), pad_dims)

        return tf.add(message_outputs, topical_outputs)
