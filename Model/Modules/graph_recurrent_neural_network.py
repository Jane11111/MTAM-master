# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 20:51
# @Author  : zxl
# @FileName: graph_recurrent_neural_network.py


import logging
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops import math_ops,array_ops,variable_scope
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell,_check_supported_dtypes,LSTMCell
from tensorflow.python.eager import context
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops import math_ops, init_ops
from keras import activations, initializers
from tensorflow.python.keras import initializers
from tensorflow.python.ops.rnn_cell_impl import RNNCell, GRUCell
from tensorflow.python.ops import math_ops, init_ops, variable_scope, array_ops, nn_ops

_BIAS_VARIABLE_NAME = 'bias'
_WEIGHTS_VARIABLE_NAME = 'kernel'

class GraphRNN():


    def build_simple_grnn_cell(self,hidden_units):
        cell  = SimpleGrnn(hidden_units)
        return MultiRNNCell([cell])

    def simple_grnn_net(self, hidden_units, input_data, input_length, scope='simple_grnn'):

        cell = self.build_simple_grnn_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs

    def build_modified_grnn_cell(self,hidden_units):
        cell  = ModifiedGrnn(hidden_units)
        return MultiRNNCell([cell])

    def modified_grnn_net(self, hidden_units, input_data, input_length, scope='modified_grnn'):

        cell = self.build_modified_grnn_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs
class ModifiedGrnn(GRUCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initialier=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(GRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        #_check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = InputSpec(ndim = 2)
        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initialier)
        self._bias_initializer = initializers.get(bias_initializer)


    def build(self, inputs_shape):
            if inputs_shape[-1] is None:
                raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                                 str(inputs_shape))
            #_check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1] - self._num_units
            self._gate_kernel = self.add_variable(
                "gates/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, 2 * self._num_units],
                initializer=self._kernel_initializer)
            self._gate_bias = self.add_variable(
                "gates/%s" % _BIAS_VARIABLE_NAME,
                shape=[2 * self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.constant_initializer(1.0, dtype=self.dtype)))
            self._candidate_kernel = self.add_variable(
                "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units  , self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))

            self._new_gate_kernel = self.add_variable(
                "new_gate/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._new_gate_bias = self.add_variable(
                "new_gate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self._w_structure = self.add_variable(
                "w_structure/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[  self._num_units],
                initializer=self._kernel_initializer)
            self._b_structure = self.add_variable(
                "b_structure/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self._w_semantic = self.add_variable(
                "w_semantic/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[2 * self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._w_semantic1 = self.add_variable(
                "w_semantic1/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[  self._num_units ],
                initializer=self._kernel_initializer)
            self._w_semantic2 = self.add_variable(
                "w_semantic2/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[ self._num_units ],
                initializer=self._kernel_initializer)
            self._b_semantic = self.add_variable(
                "b_semantic/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self._w1 = self.add_variable(
                "w1/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[ self._num_units],
                initializer=self._kernel_initializer)
            self._b1 = self.add_variable(
                "b1/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self._w2 = self.add_variable(
                "w2/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[ self._num_units],
                initializer=self._kernel_initializer)
            self._b2 = self.add_variable(
                "b2/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))

    def cosine(self,q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name='scores_cal')
        return score


    def call(self, inputs, state):
        dtype = inputs.dtype
        time_now_score = tf.expand_dims(inputs[:, -1], -1)
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        # inputs = inputs[:, :-2]
        neighbors = inputs[:,-self._num_units:]
        inputs = inputs[:,:self._num_units]
        input_size = inputs.get_shape().with_rank(2)[1]
        # decay gates
        scope = variable_scope.get_variable_scope()


        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs,  r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)


        # v9
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state,neighbors], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # c = self._activation(candidate)



        # v1
        new_u = math_ops.matmul(
            array_ops.concat([inputs,neighbors],1), self._new_gate_kernel )
        new_u = nn_ops.bias_add(new_u, self._new_gate_bias)
        new_u = math_ops.sigmoid(new_u)


        # v8
        # new_u = tf.reshape(self.cosine(inputs, neighbors), [-1,1])
        # u*=new_u

        # v2
        # struct_feature = nn_ops.bias_add(math_ops.matmul(neighbors,self._w_structure),self._b_structure)
        # semantic_feature = nn_ops.bias_add(math_ops.matmul(array_ops.concat([inputs,state],1),self._w_semantic),
        #                                    self._b_semantic)
        # new_u = math_ops.sigmoid(math_ops.matmul(struct_feature,self._w1)+
        #                          math_ops.matmul(semantic_feature,self._w2)+
        #                          self._b1)

        # v3
        # struct_feature =  tf.nn.relu(neighbors*self._w_structure+self._b_structure)
        # semantic_feature =   tf.nn.relu(inputs* self._w_semantic1+state *self._w_semantic2+self._b_semantic)
        # new_u = math_ops.sigmoid (struct_feature*self._w1 +
        #                           semantic_feature*self._w2 + self._b1)


        new_h = u * state   + (1 - u) * c *new_u

        return new_h, new_h

class SimpleGrnn(GRUCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initialier=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(GRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        #_check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = InputSpec(ndim = 2)
        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initialier)
        self._bias_initializer = initializers.get(bias_initializer)


    def build(self, inputs_shape):
            if inputs_shape[-1] is None:
                raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                                 str(inputs_shape))
            #_check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1]
            self._gate_kernel = self.add_variable(
                "gates/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, 2 * self._num_units],
                initializer=self._kernel_initializer)
            self._gate_bias = self.add_variable(
                "gates/%s" % _BIAS_VARIABLE_NAME,
                shape=[2 * self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.constant_initializer(1.0, dtype=self.dtype)))
            self._candidate_kernel = self.add_variable(
                "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units , self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
    def call(self, inputs, state):
        dtype = inputs.dtype
        time_now_score = tf.expand_dims(inputs[:, -1], -1)
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        # inputs = inputs[:, :-2]
        # neighbors = inputs[:,-self._num_units:]
        # inputs = inputs[:,:self._num_units]
        input_size = inputs.get_shape().with_rank(2)[1]
        # decay gates
        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                #weights for time now
                self._time_kernel_w1 = variable_scope.get_variable(
                    "_time_kernel_w1", shape=[self._num_units], dtype=dtype)
                self._time_kernel_b1 = variable_scope.get_variable(
                    "_time_kernel_b1", shape=[self._num_units], dtype=dtype)
                self._time_history_w1 =variable_scope.get_variable(
                    "_time_history_w1", shape=[self._num_units], dtype=dtype)
                self._time_history_b1 =variable_scope.get_variable(
                    "_time_history_b1", shape=[self._num_units], dtype=dtype)
                self._time_w1 = variable_scope.get_variable(
                    "_time_w1", shape=[self._num_units], dtype=dtype)
                self._time_w12 = variable_scope.get_variable(
                    "_time_w12", shape=[self._num_units], dtype=dtype)
                self._time_b1 = variable_scope.get_variable(
                    "_time_b1", shape=[self._num_units], dtype=dtype)
                self._time_b12 = variable_scope.get_variable(
                    "_time_b12", shape=[self._num_units], dtype=dtype)
                #weight for time last
                self._time_kernel_w2 = variable_scope.get_variable(
                    "_time_kernel_w2", shape=[self._num_units], dtype=dtype)
                self._time_kernel_b2 = variable_scope.get_variable(
                    "_time_kernel_b2", shape=[self._num_units], dtype=dtype)
                self._time_history_w2 =variable_scope.get_variable(
                    "_time_history_w2", shape=[self._num_units], dtype=dtype)
                self._time_history_b2 =variable_scope.get_variable(
                    "_time_history_b2", shape=[self._num_units], dtype=dtype)
                self._time_w2 = variable_scope.get_variable(
                    "_time_w2", shape=[self._num_units], dtype=dtype)
                self._time_b2 = variable_scope.get_variable(
                    "_time_b2", shape=[self._num_units], dtype=dtype)
                self._neighbor_w = variable_scope.get_variable(
                    "_neighbor_w", shape=[self._num_units], dtype=dtype)
                self._neighbor_b = variable_scope.get_variable(
                    "_neighbor_b", shape=[self._num_units], dtype=dtype)

        # time_last_weight = tf.nn.relu(inputs * self._time_kernel_w1 + self._time_kernel_b1+state * self._time_history_w1)
        # time_last_score = tf.nn.relu(self._time_w1 * time_last_score+ self._time_b1)
        # time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_last_score+self._time_b12)


        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs,  r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)

        new_h = u * state + (1 - u) * c

        return new_h, new_h