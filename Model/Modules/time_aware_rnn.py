from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from keras import activations, initializers
import tensorflow as tf
from tensorflow import sigmoid
from keras.activations import sigmoid
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras import initializers
#from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, _check_supported_dtypes,_hasattr
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops, init_ops, variable_scope, array_ops, nn_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell, GRUCell,_zero_state_tensors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops


_BIAS_VARIABLE_NAME = 'bias'
_WEIGHTS_VARIABLE_NAME = 'kernel'


class TimeAwareGRUCell_sigmoid(GRUCell):
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
            input_depth = inputs_shape[-1]-2
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
                shape=[input_depth + self._num_units, self._num_units],
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
        inputs = inputs[:, :-2]
        input_size = inputs.get_shape().with_rank(2)[1]
        # decay gates
        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                self._time_input_w1 = variable_scope.get_variable(
                    "_time_input_w1", shape=[self._num_units], dtype=dtype)
                self._time_input_bias1 = variable_scope.get_variable(
                    "_time_input_bias1", shape=[self._num_units], dtype=dtype)
                self._time_input_w2 = variable_scope.get_variable(
                    "_time_input_w2", shape=[self._num_units], dtype=dtype)
                self._time_input_bias2 = variable_scope.get_variable(
                    "_time_input_bias2", shape=[self._num_units], dtype=dtype)
                self._time_kernel_w1 = variable_scope.get_variable(
                    "_time_kernel_w1", shape=[input_size, self._num_units], dtype=dtype)
                self._time_kernel_t1 = variable_scope.get_variable(
                    "_time_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
                self._time_bias1 = variable_scope.get_variable(
                    "_time_bias1", shape=[self._num_units], dtype=dtype)
                self._time_kernel_w2 = variable_scope.get_variable(
                    "_time_kernel_w2", shape=[input_size, self._num_units], dtype=dtype)
                self._time_kernel_t2 = variable_scope.get_variable(
                    "_time_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
                self._time_bias2 = variable_scope.get_variable(
                    "_time_bias2", shape=[self._num_units], dtype=dtype)
                #self._o_kernel_t1 = variable_scope.get_variable(
                    #"_o_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
                #self._o_kernel_t2 = variable_scope.get_variable(
                    #"_o_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
        #time_now_input = tf.nn.tanh(tf.log(1+time_now_score) * self._time_input_w1 + self._time_input_bias1)
        #time_last_input = tf.nn.tanh(tf.log(1+time_last_score) * self._time_input_w2 + self._time_input_bias2)
        time_now_input = tf.nn.tanh(time_now_score * self._time_input_w1 + self._time_input_bias1)
        time_last_input = tf.nn.tanh(time_last_score * self._time_input_w2 + self._time_input_bias2)

        time_now_state = math_ops.matmul(inputs, self._time_kernel_w1) + \
                         math_ops.matmul(time_now_input,self._time_kernel_t1) + self._time_bias1
        time_last_state = math_ops.matmul(inputs, self._time_kernel_w2) + \
                          math_ops.matmul(time_last_input,self._time_kernel_t2) + self._time_bias2

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #new_h = u * state * sigmoid(time_last_state) + (1 - u) * c * sigmoid(time_now_state)
        new_h = u * state * sigmoid(time_now_state) + (1 - u) * c * sigmoid(time_last_state)
        return new_h, new_h

class TimeAwareGRUCell_for_gnn(GRUCell):
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
            input_depth = inputs_shape[-1]-2
            self._gate_kernel = self.add_variable(
                "gates/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units  , 2 * self._num_units],
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
        inputs = inputs[:, :-2]
        input_size = inputs.get_shape().with_rank(2)[1]
        # decay gates
        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                #weights for time now
                self._time_kernel_w1 = variable_scope.get_variable(
                    "_time_kernel_w1", shape=[self._num_units *2 , self._num_units ], dtype=dtype)
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

        #time_now_weight = tf.nn.relu( inputs * self._time_kernel_w1+self._time_kernel_b1)
        time_last_weight = tf.nn.relu(tf.matmul(inputs , self._time_kernel_w1) + self._time_kernel_b1+state * self._time_history_w1)
        #time_now_state = tf.sigmoid( time_now_weight+ self._time_w1*tf.log(time_now_score+1)+self._time_b12)

        #time_last_weight =  tf.nn.relu(inputs* self._time_kernel_w2+self._time_kernel_b2 +state * self._time_history_w2)
        #time_last_state = tf.sigmoid( time_last_weight+ self._time_w2*tf.log(time_last_score+1)+self._time_b2)

        #version 2
        #time_last_score =  tf.nn.relu(self._time_w1 * tf.log(time_last_score + 1) + self._time_b1)
        time_last_score = tf.nn.relu(self._time_w1 * time_last_score+ self._time_b1)
        time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_last_score+self._time_b12)
        #time_last_score = tf.nn.relu(self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)




        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #time_last_weight = tf.nn.relu(inputs * self._time_kernel_w2 + self._time_kernel_b2 + state * self._time_history_w2)
        #time_last_state = tf.sigmoid(time_last_weight + self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)
        #new_h = u * state *  time_now_state + (1 - u) * c * time_last_state 0.0185 0.0136
        #new_h = u * state + (1 - u) * c * time_now_state 0.0237 0.013
        #new_h = u * state * time_now_state + (1 - u) * c * time_last_state #no position 0.0211 0.0137
        #new_h = u * state + (1 - u) * c * time_now_state #no position 0.0211 0.0143
        #new_h = u * state + (1 - u) * c 0.0185 0.0138
        #####
        #sli_rec no position 0.026 0.0144
        #####
        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = u * state + (1 - u) * c * time_last_state
        return new_h, new_h


class TimeAwareGRUCell_decay_new(GRUCell):
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
            input_depth = inputs_shape[-1]-2
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
                shape=[input_depth + self._num_units, self._num_units],
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
        inputs = inputs[:, :-2]
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

        #time_now_weight = tf.nn.relu( inputs * self._time_kernel_w1+self._time_kernel_b1)
        time_last_weight = tf.nn.relu(inputs * self._time_kernel_w1 + self._time_kernel_b1+state * self._time_history_w1)
        #time_now_state = tf.sigmoid( time_now_weight+ self._time_w1*tf.log(time_now_score+1)+self._time_b12)

        #time_last_weight =  tf.nn.relu(inputs* self._time_kernel_w2+self._time_kernel_b2 +state * self._time_history_w2)
        #time_last_state = tf.sigmoid( time_last_weight+ self._time_w2*tf.log(time_last_score+1)+self._time_b2)

        #version 2
        #time_last_score =  tf.nn.relu(self._time_w1 * tf.log(time_last_score + 1) + self._time_b1)
        time_last_score = tf.nn.relu(self._time_w1 * time_last_score+ self._time_b1)
        time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_last_score+self._time_b12)
        #time_last_score = tf.nn.relu(self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)




        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #time_last_weight = tf.nn.relu(inputs * self._time_kernel_w2 + self._time_kernel_b2 + state * self._time_history_w2)
        #time_last_state = tf.sigmoid(time_last_weight + self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)
        #new_h = u * state *  time_now_state + (1 - u) * c * time_last_state 0.0185 0.0136
        #new_h = u * state + (1 - u) * c * time_now_state 0.0237 0.013
        #new_h = u * state * time_now_state + (1 - u) * c * time_last_state #no position 0.0211 0.0137
        #new_h = u * state + (1 - u) * c * time_now_state #no position 0.0211 0.0143
        #new_h = u * state + (1 - u) * c 0.0185 0.0138
        #####
        #sli_rec no position 0.026 0.0144
        #####
        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = u * state + (1 - u) * c * time_last_state
        return new_h, new_h


class TimeAwareGRUCell_Extend(GRUCell):
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
            input_depth = inputs_shape[-1]-2
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
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
    def call(self, inputs, state):
        dtype = inputs.dtype
        time_next_score = tf.expand_dims(inputs[:, -1], -1)
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]
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

        #time_now_weight = tf.nn.relu( inputs * self._time_kernel_w1+self._time_kernel_b1)
        time_last_weight = tf.nn.relu(inputs * self._time_kernel_w1 + self._time_kernel_b1+state * self._time_history_w1)
        #time_now_state = tf.sigmoid( time_now_weight+ self._time_w1*tf.log(time_now_score+1)+self._time_b12)

        #time_last_weight =  tf.nn.relu(inputs* self._time_kernel_w2+self._time_kernel_b2 +state * self._time_history_w2)
        #time_last_state = tf.sigmoid( time_last_weight+ self._time_w2*tf.log(time_last_score+1)+self._time_b2)

        #version 2
        #time_last_score =  tf.nn.relu(self._time_w1 * tf.log(time_last_score + 1) + self._time_b1)
        time_next_score = tf.nn.relu(self._time_w1 * time_next_score+ self._time_b1)
        time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_next_score+self._time_b12)
        #time_last_score = tf.nn.relu(self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)




        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #time_last_weight = tf.nn.relu(inputs * self._time_kernel_w2 + self._time_kernel_b2 + state * self._time_history_w2)
        #time_last_state = tf.sigmoid(time_last_weight + self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)
        #new_h = u * state *  time_now_state + (1 - u) * c * time_last_state 0.0185 0.0136
        #new_h = u * state + (1 - u) * c * time_now_state 0.0237 0.013
        #new_h = u * state * time_now_state + (1 - u) * c * time_last_state #no position 0.0211 0.0137
        #new_h = u * state + (1 - u) * c * time_now_state #no position 0.0211 0.0143
        #new_h = u * state + (1 - u) * c 0.0185 0.0138
        #####
        #sli_rec no position 0.026 0.0144
        #####
        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = u * state + (1 - u) * c * time_last_state
        return new_h, new_h


class TimePredictionGRUCell_old(GRUCell):
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
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = input_spec.InputSpec(ndim = 2)
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
            _check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1]-2
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
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self.output_w = self.add_variable(
                "output_w",
                shape=[self._num_units,self._num_units-1],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
    def call(self, inputs, state):
        dtype = inputs.dtype
        avg_interval = inputs[:, -1]
        time_last = inputs[:, -2]
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]

        interval = state[:,-1] # 预测的时间
        state = math_ops.matmul(state[:,:-1],self.output_w,transpose_b=True)
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


                self._new_time_w1 = variable_scope.get_variable(
                    "_new_time_w1", shape=[self._num_units], dtype=dtype)
                self._new_time_b1 = variable_scope.get_variable(
                    "_new_time_b1", shape=[self._num_units], dtype=dtype)
                self._t_w1 = variable_scope.get_variable(
                    "_t_w1", shape=[1], dtype=dtype)
                self._t_w11 = variable_scope.get_variable(
                    "_t_w11", shape=[self._num_units], dtype=dtype)
                self._t_w12 = variable_scope.get_variable(
                    "_t_w12", shape=[self._num_units], dtype=dtype)
                self._t_w13 = variable_scope.get_variable(
                    "_t_w13", shape=[self._num_units], dtype=dtype)
                self._t_b1 = variable_scope.get_variable(
                    "_t_b1", shape=[1], dtype=dtype)
                self._t_w2 = variable_scope.get_variable(
                    "_t_w2", shape=[1], dtype=dtype)
                self._t_b2 = variable_scope.get_variable(
                    "_t_b2", shape=[1], dtype=dtype)
                self._t_w3 = variable_scope.get_variable(
                    "_t_w3", shape=[1], dtype=dtype)
                self._t_b3 = variable_scope.get_variable(
                    "_t_b3", shape=[1], dtype=dtype)
                self._t_w4 = variable_scope.get_variable(
                    "_t_w4", shape=[1], dtype=dtype)
                self._t_b4 = variable_scope.get_variable(
                    "_t_b4", shape=[1], dtype=dtype)
                self._t_w5 = variable_scope.get_variable(
                    "_t_w5", shape=[1], dtype=dtype)
                self._t_w6 = variable_scope.get_variable(
                    "_t_w6", shape=[1], dtype=dtype)
                self._t_w7 = variable_scope.get_variable(
                    "_t_w7", shape=[1], dtype=dtype)
                self._t_w8 = variable_scope.get_variable(
                    "_t_w8", shape=[1], dtype=dtype)
                self._t_w9 = variable_scope.get_variable(
                    "_t_w9", shape=[1], dtype=dtype)
                self._t_w10 = variable_scope.get_variable(
                    "_t_w10", shape=[1], dtype=dtype)

        """
        以下是预测的下一个时间
        这个时间是对下一个行为预测的补充
        """

        # v2
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(
        #         inputs * self._t_w11 + state * self._t_w12) + time_last * self._t_w2 +  self._t_b2)
        # new_t = interval * self._t_w1 +candidate_interval * self._t_w3 + self._t_b3  # batch_size,

        # v1
        # r_t = math_ops.sigmoid(tf.reduce_mean(
        #     inputs * self._t_w11) + interval * self._t_w1 + time_last * self._t_w4 + avg_interval * self._t_w5 + self._t_b1)
        # z_t = math_ops.sigmoid(tf.reduce_mean(
        #     inputs * self._t_w12) + interval * self._t_w2 + time_last * self._t_w6 + avg_interval * self._t_w7 + self._t_b2)
        # candidate_interval = tf.nn.relu(r_t * interval + tf.reduce_mean(
        #     inputs * self._t_w3) + time_last * self._t_w8 + avg_interval * self._t_w9 + self._t_b3)
        # new_t = z_t * candidate_interval + (1 - z_t) * interval

        #v3

        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11) + time_last * self._t_w2 + interval * self._t_w1 + self._t_b2)
        # new_t = candidate_interval  # batch_size,

        #v4
        # b = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1+ self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + avg_interval * self._t_w3 + self._t_b2)
        # new_t = b * candidate_interval + (1 - b) * avg_interval  # batch_size,
        #
        #
        # new_reset_gate = tf.nn.sigmoid(self._new_time_w1*tf.expand_dims(new_t,-1)+self._new_time_b1)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        # # TODO modified
        # r = r*new_reset_gate

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)

        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = math_ops.matmul((u * state + (1 - u) * c),self.output_w)

        """
        version 5
        """
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11) + time_last * self._t_w2 + interval * self._t_w1 + self._t_b2)
        # new_t = candidate_interval  # batch_size,

        """
        version 17
        """
        # b = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1+ self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + avg_interval * self._t_w3 + self._t_b2)
        # new_t = b * candidate_interval + (1 - b) * avg_interval  # batch_size,

        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)


        """
        version 24
        """
        r_t = math_ops.sigmoid(tf.reduce_mean(
            inputs * self._t_w11) + interval * self._t_w1 + time_last * self._t_w4 + avg_interval * self._t_w5 + self._t_b1)
        z_t = math_ops.sigmoid(tf.reduce_mean(
            inputs * self._t_w12) + interval * self._t_w2 + time_last * self._t_w6 + avg_interval * self._t_w7 + self._t_b2)
        candidate_interval = tf.nn.relu(r_t * interval + tf.reduce_mean(
            inputs * self._t_w3) + time_last * self._t_w8 + avg_interval * self._t_w9 + self._t_b3)
        new_t = z_t * candidate_interval + (1 - z_t) * interval


        new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        # b = math_ops.sigmoid(
        #     time_last * self._t_w1 + tf.reduce_mean(inputs * self._t_w11 + state * self._t_w12, 1) + self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + self._t_b2)
        # new_t = b * interval + (1 - b) * candidate_interval
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        return new_h, new_h


class TimePredictionGRUCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(TimePredictionGRUCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return  2 *  self._num_units +1

    @property
    def output_size(self):
        return  2 *  self._num_units +1

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value -2
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_gate_kernel = self.add_variable(
            "_time_gate_kernel",
            shape=[3 * self._num_units, 2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_gate_bias = self.add_variable(
            "_time_gate_bias",
            shape=[2 * self._num_units],
            initializer=self._kernel_initializer)
        self._time_candidate_kernel = self.add_variable(
            "_time_candidate_kernel",
            shape=[3 * self._num_units, self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_candidate_bias = self.add_variable(
            "_time_candidate_bias",
            shape=[self._num_units],
            initializer=self._kernel_initializer)

        self.built = True

    def call(self,inputs, state):
        dtype = inputs.dtype
        avg_interval = inputs[:, -1]
        time_last = inputs[:, -2]
        time_last_score = tf.expand_dims(time_last,axis=-1)

        inputs = inputs[:, :self._num_units]

        time_state = state[:, self._num_units:-1]  # 预测的时间
        interval = state[:,-1]
        state = state[:, :self._num_units]

        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                # weights for time now
                # self._gate_kernel = variable_scope.get_variable(
                #     "_gate_kernel", shape=[2*self._num_units,2*self._num_units], dtype=dtype)
                # self._gate_bias = variable_scope.get_variable(
                #     "_gate_bias", shape=[2 * self._num_units ], dtype=dtype)

                # self._candidate_kernel = variable_scope.get_variable(
                #     "_candidate_kernel", shape=[ 2 * self._num_units,  self._num_units], dtype=dtype)
                # self._candidate_bias = variable_scope.get_variable(
                #     "_candidate_bias", shape=[ self._num_units], dtype=dtype)

                self._time_pred_w = variable_scope.get_variable(
                    "_time_pred_w", shape=[self._num_units], dtype=dtype)
                self._time_pred_w2 = variable_scope.get_variable(
                    "_time_pred_w2", shape=[self._num_units], dtype=dtype)
                self._time_pred_b = variable_scope.get_variable(
                    "_time_pred_b", shape=[1], dtype=dtype)

                self._time_w1 = variable_scope.get_variable(
                    "_time_w1", shape=[self._num_units], dtype=dtype)
                self._time_b1 = variable_scope.get_variable(
                    "_time_b1", shape=[self._num_units], dtype=dtype)
                self._time_w2 = variable_scope.get_variable(
                    "_time_w2", shape=[self._num_units], dtype=dtype)
                self._time_b2 = variable_scope.get_variable(
                    "_time_b2", shape=[self._num_units], dtype=dtype)

                self._time_w3 = variable_scope.get_variable(
                    "_time_w3", shape=[self._num_units], dtype=dtype)
                self._time_b3 = variable_scope.get_variable(
                    "_time_b3", shape=[self._num_units], dtype=dtype)
                self._time_w4 = variable_scope.get_variable(
                    "_time_w4", shape=[self._num_units], dtype=dtype)
                self._time_b4 = variable_scope.get_variable(
                    "_time_b4", shape=[self._num_units], dtype=dtype)
                self._time_w5 = variable_scope.get_variable(
                    "_time_w5", shape=[self._num_units], dtype=dtype)
                self._time_w6 = variable_scope.get_variable(
                    "_time_w6", shape=[self._num_units], dtype=dtype)
                self._time_b5 = variable_scope.get_variable(
                    "_time_b5", shape=[self._num_units], dtype=dtype)
                self._semantic_span_w1 = variable_scope.get_variable(
                    "_semantic_span_w1", shape=[self._num_units], dtype=dtype)
                self._semantic_span_b1 = variable_scope.get_variable(
                    "_semantic_span_b1", shape=[self._num_units], dtype=dtype)
                self._semantic_span_w2 = variable_scope.get_variable(
                    "_semantic_span_w2", shape=[self._num_units], dtype=dtype)
                self._semantic_span_b2 = variable_scope.get_variable(
                    "_semantic_span_b2", shape=[self._num_units], dtype=dtype)
                self._semantic_last_w1 = variable_scope.get_variable(
                    "_semantic_last_w1", shape=[self._num_units], dtype=dtype)
                self._semantic_last_b1 = variable_scope.get_variable(
                    "_semantic_last_b1", shape=[self._num_units], dtype=dtype)
                self._semantic_last_w2 = variable_scope.get_variable(
                    "_semantic_last_w2", shape=[self._num_units], dtype=dtype)
                self._semantic_last_b2 = variable_scope.get_variable(
                    "_semantic_last_b2", shape=[self._num_units], dtype=dtype)
                self._time_span_w1 = variable_scope.get_variable(
                    "_time_span_w1", shape=[self._num_units], dtype=dtype)
                self._time_span_b1 = variable_scope.get_variable(
                    "_time_span_b1", shape=[self._num_units], dtype=dtype)
                self._time_last_w1 = variable_scope.get_variable(
                    "_time_last_w1", shape=[self._num_units], dtype=dtype)
                self._time_last_b1 = variable_scope.get_variable(
                    "_time_last_b1", shape=[self._num_units], dtype=dtype)
                self._t_w11 = variable_scope.get_variable(
                    "_t_w11", shape=[self._num_units], dtype=dtype)
                self._t_w12 = variable_scope.get_variable(
                    "_t_w12", shape=[self._num_units], dtype=dtype)
                self._t_w13 = variable_scope.get_variable(
                    "_t_w13", shape=[self._num_units], dtype=dtype)
                self._t_w14 = variable_scope.get_variable(
                    "_t_w14", shape=[self._num_units], dtype=dtype)
                self._t_w15 = variable_scope.get_variable(
                    "_t_w15", shape=[self._num_units], dtype=dtype)
                self._t_w16 = variable_scope.get_variable(
                    "_t_w16", shape=[self._num_units], dtype=dtype)
                self._t_b11 = variable_scope.get_variable(
                    "_t_b11", shape=[self._num_units], dtype=dtype)
                self._t_b12 = variable_scope.get_variable(
                    "_t_b12", shape=[self._num_units], dtype=dtype)
                self._t_b13 = variable_scope.get_variable(
                    "_t_b13", shape=[self._num_units], dtype=dtype)
                self._t_w1 = variable_scope.get_variable(
                    "_t_w1", shape=[1], dtype=dtype)
                self._t_b1 = variable_scope.get_variable(
                    "_t_b1", shape=[1], dtype=dtype)
                self._t_w2 = variable_scope.get_variable(
                    "_t_w2", shape=[1], dtype=dtype)
                self._t_b2 = variable_scope.get_variable(
                    "_t_b2", shape=[1], dtype=dtype)
                self._t_w3 = variable_scope.get_variable(
                    "_t_w3", shape=[1], dtype=dtype)
                self._t_b3 = variable_scope.get_variable(
                    "_t_b3", shape=[1], dtype=dtype)
                self._t_w4 = variable_scope.get_variable(
                    "_t_w4", shape=[1], dtype=dtype)
                self._t_b4 = variable_scope.get_variable(
                    "_t_b4", shape=[1], dtype=dtype)
                self._t_w5 = variable_scope.get_variable(
                    "_t_w5", shape=[1], dtype=dtype)
                self._t_w6 = variable_scope.get_variable(
                    "_t_w6", shape=[1], dtype=dtype)
                self._t_w7 = variable_scope.get_variable(
                    "_t_w7", shape=[1], dtype=dtype)
                self._t_w8 = variable_scope.get_variable(
                    "_t_w8", shape=[1], dtype=dtype)
                self._t_w9 = variable_scope.get_variable(
                    "_t_w9", shape=[1], dtype=dtype)
                self._t_w10 = variable_scope.get_variable(
                    "_t_w10", shape=[1], dtype=dtype)
        """
        v7
        """
        """
        # gru预测新的时间
        

        # TODO 预测时间用的feature与计算gate用的feature需要分开吗
        time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units

        time_gate_inputs = math_ops.matmul(
            array_ops.concat([inputs,time_state,time_feature],1),self._time_gate_kernel
        )
        time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        time_value = math_ops.sigmoid(time_gate_inputs)

        t_r,t_u = array_ops.split(value = time_value, num_or_size_splits=2,axis=1)
        time_reset_state = t_r * time_state

        time_candidate = math_ops.matmul(
            array_ops.concat([inputs,time_reset_state,time_feature],1),self._time_candidate_kernel
        )
        time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        time_candidate = self._activation(time_candidate)

        new_time_state = t_u * time_state + (1-t_u) * time_candidate

        new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w,axis=1) +self._time_pred_b)



        # temporary gate

        # semantic_last_feature  = tf.nn.relu(
        #     inputs * self._semantic_last_w1+ state * self._semantic_last_w2 + self._semantic_last_b1 )
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 + state * self._semantic_span_w2 + self._semantic_span_b1)
        time_feature2 = tf.nn.relu(self._t_w12 * time_last_score + self._t_b12)
        semantic_last_feature = tf.nn.relu(
            inputs * self._semantic_last_w1  + self._semantic_last_b1)
        semantic_span_feature = tf.nn.relu(
            inputs * self._semantic_span_w1  + self._semantic_span_b1)

        time_last_feature = tf.nn.relu(time_feature2 * self._time_last_w1 + self._time_last_b1)
        time_span_feature = tf.nn.relu(new_time_state * self._time_span_w1 + self._time_span_b1)

        # 计算 time aware gate
        time_last_state = tf.sigmoid(
            self._time_w4 * semantic_last_feature +
            self._time_w5 * time_last_feature +
            self._time_b3)
        time_span_state = tf.sigmoid(
            self._t_w13 * semantic_span_feature +
            self._t_w14 * time_span_feature +
            self._t_b13)




        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        value = math_ops.sigmoid(gate_inputs)

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        candidate = self._activation(candidate)

        # new_h = u * state + (1 - u) * c * time_last_state
        # new_h = (u * state + (1 - u) * c)
        new_h = u * time_last_state * state + (1-u) * time_span_state * candidate

        new_h = array_ops.concat([new_h, new_time_state,tf.expand_dims(new_time_value, -1)], 1)
        """
        """
        v8
        """
        # gru预测新的时间
        """
        time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units

        time_gate_inputs = math_ops.matmul(
            array_ops.concat([time_state, time_feature], 1), self._time_gate_kernel
        )
        time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        time_value = math_ops.sigmoid(time_gate_inputs)

        t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        time_reset_state = t_r * time_state

        time_candidate = math_ops.matmul(
            array_ops.concat([time_reset_state, time_feature], 1), self._time_candidate_kernel
        )
        time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        time_candidate = self._activation(time_candidate)

        new_time_state = t_u * time_state + (1 - t_u) * time_candidate

        new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
                                    self._time_pred_b)

        # temporary gate

        # semantic_last_feature  = tf.nn.relu(
        #     inputs * self._semantic_last_w1+ state * self._semantic_last_w2 + self._semantic_last_b1 )
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 + state * self._semantic_span_w2 + self._semantic_span_b1)
        time_feature2 = tf.nn.relu(self._t_w12 * time_last_score + self._t_b12)
        semantic_last_feature = tf.nn.relu(
            inputs * self._semantic_last_w1 + self._semantic_last_b1)
        semantic_span_feature = tf.nn.relu(
            inputs * self._semantic_span_w1 + self._semantic_span_b1)

        time_last_feature = tf.nn.relu(time_feature2 * self._time_last_w1 + self._time_last_b1)
        time_span_feature = tf.nn.relu(new_time_state * self._time_span_w1 + self._time_span_b1)

        # 计算 time aware gate
        time_last_state = tf.sigmoid(
            self._time_w4 * semantic_last_feature +
            self._time_w5 * time_last_feature +
            self._time_b3)
        time_span_state = tf.sigmoid(
            self._t_w13 * semantic_span_feature +
            self._t_w14 * time_span_feature +
            self._t_b13)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        value = math_ops.sigmoid(gate_inputs)

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        candidate = self._activation(candidate)

        # new_h = u * state + (1 - u) * c * time_last_state
        # new_h = (u * state + (1 - u) * c)
        new_h = u * time_last_state * state + (1 - u) * time_span_state * candidate

        new_h = array_ops.concat([new_h, new_time_state, tf.expand_dims(new_time_value, -1)], 1)
        """
        """
        v9
        """
        """
        # gru预测新的时间

        time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units

        time_gate_inputs = math_ops.matmul(
            array_ops.concat([inputs,time_state, time_feature], 1), self._time_gate_kernel
        )
        time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        time_value = math_ops.sigmoid(time_gate_inputs)

        t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        time_reset_state = t_r * time_state

        time_candidate = math_ops.matmul(
            array_ops.concat([inputs,time_reset_state, time_feature], 1), self._time_candidate_kernel
        )
        time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        time_candidate = self._activation(time_candidate)

        new_time_state = t_u * time_state + (1 - t_u) * time_candidate

        new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
                                    tf.reduce_mean(inputs*self._time_pred_w2,axis=1)+
                                    self._t_w1 * time_last +
                                    self._time_pred_b)

        # temporary gate

        # semantic_last_feature  = tf.nn.relu(
        #     inputs * self._semantic_last_w1+ state * self._semantic_last_w2 + self._semantic_last_b1 )
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 + state * self._semantic_span_w2 + self._semantic_span_b1)
        time_feature2 = tf.nn.relu(self._t_w12 * time_last_score + self._t_b12)
        semantic_last_feature = tf.nn.relu(
            inputs * self._semantic_last_w1 + self._semantic_last_b1)
        semantic_span_feature = tf.nn.relu(
            inputs * self._semantic_span_w1 + self._semantic_span_b1)

        time_last_feature = tf.nn.relu(time_feature2 * self._time_last_w1 + self._time_last_b1)
        time_span_feature = tf.nn.relu(new_time_state * self._time_span_w1 + self._time_span_b1)

        # 计算 time aware gate
        time_last_state = tf.sigmoid(
            self._time_w4 * semantic_last_feature +
            self._time_w5 * time_last_feature +
            self._time_b3)
        time_span_state = tf.sigmoid(
            self._t_w13 * semantic_span_feature +
            self._t_w14 * time_span_feature +
            self._t_b13)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        value = math_ops.sigmoid(gate_inputs)

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        candidate = self._activation(candidate)

        # new_h = u * state + (1 - u) * c * time_last_state
        # new_h = (u * state + (1 - u) * c)
        new_h = u * time_last_state * state + (1 - u) * time_span_state * candidate
        """

        """
        v11
        """
        # time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units
        #
        # time_gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, time_state, time_feature], 1), self._time_gate_kernel
        # )
        # time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        # time_value = math_ops.sigmoid(time_gate_inputs)
        #
        # t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        # time_reset_state = t_r * time_state
        #
        # time_candidate = math_ops.matmul(
        #     array_ops.concat([inputs, time_reset_state, time_feature], 1), self._time_candidate_kernel
        # )
        # time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        # time_candidate = self._activation(time_candidate)
        #
        # new_time_state = t_u * time_state + (1 - t_u) * time_candidate
        #
        # new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
        #                             tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
        #                             self._t_w1 * time_last +
        #                             self._time_pred_b)
        # time_span_score = tf.expand_dims(new_time_value, axis=-1)
        #
        # # temporary gate
        #
        # semantic_last_feature = tf.nn.relu(
        #     inputs * self._semantic_last_w1 + state * self._semantic_last_w2 + self._semantic_last_b1)
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 + state * self._semantic_span_w2 + self._semantic_span_b1)
        #
        # time_last_feature = tf.nn.relu(time_last_score * self._time_last_w1 + self._time_last_b1)
        # time_span_feature = tf.nn.relu(time_span_score * self._time_span_w1 + self._time_span_b1)
        #
        # # 计算 time aware gate
        # time_last_state = tf.sigmoid(
        #     self._time_w4 * semantic_last_feature +
        #     self._time_w5 * time_last_feature +
        #     self._time_b3)
        # time_span_state = tf.sigmoid(
        #     self._t_w13 * semantic_span_feature +
        #     self._t_w14 * time_span_feature +
        #     self._t_b13)
        # # time_last_weight = tf.nn.relu(
        # #     inputs * self._time_kernel_w1 + self._time_kernel_b1 + state * self._time_history_w1)
        # #
        # # time_last_score = tf.nn.relu(self._time_w1 * time_last_score + self._time_b1)
        # # time_last_state = tf.sigmoid(
        # #     self._time_kernel_w2 * time_last_weight + self._time_w12 * time_last_score + self._time_b12)
        #
        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        # value = math_ops.sigmoid(gate_inputs)
        #
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        # r_state = r * state
        #
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # candidate = self._activation(candidate)
        #
        # # new_h = u * state + (1 - u) * c * time_last_state
        # # new_h = (u * state + (1 - u) * c)
        # new_h = u * state * time_last_state + (1 - u) * candidate * time_span_state

        """
        v10
        """

        # time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units
        #
        # time_gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, time_state, time_feature], 1), self._time_gate_kernel
        # )
        # time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        # time_value = math_ops.sigmoid(time_gate_inputs)
        #
        # t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        # time_reset_state = t_r * time_state
        #
        # time_candidate = math_ops.matmul(
        #     array_ops.concat([inputs, time_reset_state, time_feature], 1), self._time_candidate_kernel
        # )
        # time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        # time_candidate = self._activation(time_candidate)
        #
        # new_time_state = t_u * time_state + (1 - t_u) * time_candidate
        #
        # new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
        #                             tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
        #                             self._t_w1 * time_last +
        #                             self._time_pred_b)
        # time_span_score = tf.expand_dims(new_time_value,axis= -1)
        #
        # # temporary gate
        #
        # semantic_last_feature = tf.nn.relu(
        #     inputs * self._semantic_last_w1 + state * self._semantic_last_w2+self._semantic_last_b1)
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 + state * self._semantic_span_w2+self._semantic_span_b1)
        #
        # time_last_feature = tf.nn.relu(time_last_score * self._time_last_w1 + self._time_last_b1)
        # time_span_feature = tf.nn.relu(time_span_score * self._time_span_w1 + self._time_span_b1)
        #
        # # 计算 time aware gate
        # time_last_state = tf.sigmoid(
        #     self._time_w4 * semantic_last_feature +
        #     self._time_w5 * time_last_feature +
        #     self._time_b3)
        # time_span_state = tf.sigmoid(
        #     self._t_w13 * semantic_span_feature +
        #     self._t_w14 * time_span_feature +
        #     self._t_b13)
        #
        #
        #
        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        # value = math_ops.sigmoid(gate_inputs)
        #
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        # r_state = r * state
        #
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # candidate = self._activation(candidate)
        #
        #
        # new_h = u * state *time_span_state + (1 - u) * candidate * time_last_state

        """
        v12
        """

        # time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units
        #
        # time_gate_inputs = math_ops.matmul(
        #     array_ops.concat([item_emb, time_state, time_feature], 1), self._time_gate_kernel
        # )
        # time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        # time_value = math_ops.sigmoid(time_gate_inputs)
        #
        # t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        # time_reset_state = t_r * time_state
        #
        # time_candidate = math_ops.matmul(
        #     array_ops.concat([item_emb, time_reset_state, time_feature], 1), self._time_candidate_kernel
        # )
        # time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        # time_candidate = self._activation(time_candidate)
        #
        # new_time_state = t_u * time_state + (1 - t_u) * time_candidate
        #
        # new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
        #                             tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
        #                             self._t_w1 * time_last +
        #                             self._time_pred_b)
        # time_span_score = tf.expand_dims(new_time_value, axis=-1)
        #
        # # temporary gate
        #
        # semantic_last_feature = tf.nn.relu(
        #     inputs * self._semantic_last_w1 + state * self._semantic_last_w2 + self._semantic_last_b1)
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 + state * self._semantic_span_w2 + self._semantic_span_b1)
        #
        # time_last_feature = tf.nn.relu(time_last_score * self._time_last_w1 + self._time_last_b1)
        # time_span_feature = tf.nn.relu(time_span_score * self._time_span_w1 + self._time_span_b1)
        #
        # # 计算 time aware gate
        # time_last_state = tf.sigmoid(
        #     self._time_w4 * semantic_last_feature +
        #     self._time_w5 * time_last_feature +
        #     self._time_b3)
        # time_span_state = tf.sigmoid(
        #     self._t_w13 * semantic_span_feature +
        #     self._t_w14 * time_span_feature +
        #     self._t_b13)
        #
        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        # value = math_ops.sigmoid(gate_inputs)
        #
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        # r_state = r * state
        #
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # candidate = self._activation(candidate)
        #
        # new_h = u * state * time_span_state + (1 - u) * candidate * time_last_state

        """
        v13
        """
        # time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units
        #
        # time_gate_inputs = math_ops.matmul(
        #     array_ops.concat([item_emb, time_state, time_feature], 1), self._time_gate_kernel
        # )
        # time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        # time_value = math_ops.sigmoid(time_gate_inputs)
        #
        # t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        # time_reset_state = t_r * time_state
        #
        # time_candidate = math_ops.matmul(
        #     array_ops.concat([item_emb, time_reset_state, time_feature], 1), self._time_candidate_kernel
        # )
        # time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        # time_candidate = self._activation(time_candidate)
        #
        # new_time_state = t_u * time_state + (1 - t_u) * time_candidate
        #
        # new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
        #                             tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
        #                             self._t_w1 * time_last +
        #                             self._time_pred_b)
        # time_span_score = tf.expand_dims(new_time_value, axis=-1)
        #
        # # temporary gate
        #
        # semantic_last_feature = tf.nn.relu(
        #     inputs * self._semantic_last_w1 + state * self._semantic_last_w2 + self._semantic_last_b1)
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 + state * self._semantic_span_w2 + self._semantic_span_b1)
        #
        # time_last_feature = tf.nn.relu(time_last_score * self._time_last_w1 + self._time_last_b1)
        # time_span_feature = tf.nn.relu(time_span_score * self._time_span_w1 + self._time_span_b1)
        #
        # # 计算 time aware gate
        # time_last_state = tf.sigmoid(
        #     self._time_w4 * semantic_last_feature +
        #     self._time_w5 * time_last_feature +
        #     self._time_b3)
        # time_span_state = tf.sigmoid(
        #     self._t_w13 * semantic_span_feature +
        #     self._t_w14 * time_span_feature +
        #     self._t_b13)
        #
        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        # value = math_ops.sigmoid(gate_inputs)
        #
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        # r_state = r * state
        #
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # candidate = self._activation(candidate)
        #
        # new_h = u * state  + (1 - u) * candidate * time_last_state

        """
        my_gru
        """
        # time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units
        #
        # time_gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, time_state, time_feature], 1), self._time_gate_kernel
        # )
        # time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        # time_value = math_ops.sigmoid(time_gate_inputs)
        #
        # t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        # time_reset_state = t_r * time_state
        #
        # time_candidate = math_ops.matmul(
        #     array_ops.concat([inputs, time_reset_state, time_feature], 1), self._time_candidate_kernel
        # )
        # time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        # time_candidate = self._activation(time_candidate)
        #
        # new_time_state = t_u * time_state + (1 - t_u) * time_candidate
        #
        # new_time_value = tf.nn.relu(tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
        #                             self._t_w1 * time_last +
        #                             self._time_pred_b)
        #
        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        #
        # value = math_ops.sigmoid(gate_inputs)
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        # r_state = r * state
        #
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # candidate = self._activation(candidate)
        #
        # new_h = u * state + (1 - u) * candidate


        """
        v14
        """

        time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units

        time_gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, time_state, time_feature], 1), self._time_gate_kernel
        )
        time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        time_value = math_ops.sigmoid(time_gate_inputs)

        t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        time_reset_state = t_r * time_state

        time_candidate = math_ops.matmul(
            array_ops.concat([inputs, time_reset_state, time_feature], 1), self._time_candidate_kernel
        )
        time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        time_candidate = self._activation(time_candidate)

        new_time_state = t_u * time_state + (1 - t_u) * time_candidate

        new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
                                    tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
                                    self._t_w1 * time_last +
                                    self._time_pred_b)
        time_span_score = tf.expand_dims(new_time_value,axis= -1)

        # temporary gate

        semantic_last_feature = tf.nn.relu(
            inputs * self._semantic_last_w1 + state * self._semantic_last_w2+self._semantic_last_b1)
        semantic_span_feature = tf.nn.relu(
            inputs * self._semantic_span_w1 + state * self._semantic_span_w2+self._semantic_span_b1)

        time_last_feature = tf.nn.relu(time_last_score * self._time_last_w1 + self._time_last_b1)
        time_span_feature = tf.nn.relu(time_span_score * self._time_span_w1 + self._time_span_b1)

        # 计算 time aware gate
        time_last_state = tf.sigmoid(
            self._time_w4 * semantic_last_feature +
            self._time_w5 * time_last_feature +
            self._time_b3)
        time_span_state = tf.sigmoid(
            self._t_w13 * semantic_span_feature +
            self._t_w14 * time_span_feature +
            self._t_b13)
        # time_last_weight = tf.nn.relu(
        #     inputs * self._time_kernel_w1 + self._time_kernel_b1 + state * self._time_history_w1)
        #
        # time_last_score = tf.nn.relu(self._time_w1 * time_last_score + self._time_b1)
        # time_last_state = tf.sigmoid(
        #     self._time_kernel_w2 * time_last_weight + self._time_w12 * time_last_score + self._time_b12)



        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        value = math_ops.sigmoid(gate_inputs)

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        candidate = self._activation(candidate)


        new_h = u * state  + (1 - u) * candidate *  time_span_state

        """
        v15
        """
        # time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units
        #
        # time_gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, time_state, time_feature], 1), self._time_gate_kernel
        # )
        # time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        # time_value = math_ops.sigmoid(time_gate_inputs)
        #
        # t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        # time_reset_state = t_r * time_state
        #
        # time_candidate = math_ops.matmul(
        #     array_ops.concat([inputs, time_reset_state, time_feature], 1), self._time_candidate_kernel
        # )
        # time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        # time_candidate = self._activation(time_candidate)
        #
        # new_time_state = t_u * time_state + (1 - t_u) * time_candidate
        #
        # new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
        #                             tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
        #                             self._t_w1 * time_last +
        #                             self._time_pred_b)
        # time_span_score = tf.expand_dims(new_time_value, axis=-1)
        #
        # # temporary gate
        #
        # semantic_last_feature = tf.nn.relu(
        #     inputs * self._semantic_last_w1 +  self._semantic_last_b1)
        # semantic_span_feature = tf.nn.relu(
        #     inputs * self._semantic_span_w1 +   self._semantic_span_b1)
        #
        # time_last_feature = tf.nn.relu(time_last_score * self._time_last_w1 + self._time_last_b1)
        # time_span_feature = tf.nn.relu(time_span_score * self._time_span_w1 + self._time_span_b1)
        #
        # # 计算 time aware gate
        # time_last_state = tf.sigmoid(
        #     self._time_w4 * semantic_last_feature +
        #     self._time_w5 * time_last_feature +
        #     self._time_b3)
        # time_span_state = tf.sigmoid(
        #     self._t_w13 * semantic_span_feature +
        #     self._t_w14 * time_span_feature +
        #     self._t_b13)
        # # time_last_weight = tf.nn.relu(
        # #     inputs * self._time_kernel_w1 + self._time_kernel_b1 + state * self._time_history_w1)
        # #
        # # time_last_score = tf.nn.relu(self._time_w1 * time_last_score + self._time_b1)
        # # time_last_state = tf.sigmoid(
        # #     self._time_kernel_w2 * time_last_weight + self._time_w12 * time_last_score + self._time_b12)
        #
        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        # value = math_ops.sigmoid(gate_inputs)
        #
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        # r_state = r * state
        #
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # candidate = self._activation(candidate)
        #
        # new_h = u * state + (1 - u) * candidate * time_span_state

        new_h = array_ops.concat([new_h, new_time_state, tf.expand_dims(new_time_value, -1)], 1)

        return new_h, new_h


class ReconsumePredictionGRUCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(ReconsumePredictionGRUCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return 2*self._num_units

    @property
    def output_size(self):
        return 2*self._num_units

    def build(self, inputs_shape):

        dtype = tf.float32

        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value - 2 - self._num_units
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._reconsume_gate_kernel = self.add_variable(
            "reconsume_gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth +  self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._reconsume_gate_bias = self.add_variable(
            "reconsume_gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._reconsume_candidate_kernel = self.add_variable(
            "reconsume_candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth +  self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._reconsume_candidate_bias = self.add_variable(
            "reconsume_candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        self._time_w1 = self.add_variable(
            "_time_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_b1 = self.add_variable(
            "_time_b1", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_w2 = self.add_variable(
            "_time_w2", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_b2 = self.add_variable(
            "_time_b2", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_w3 = self.add_variable(
            "_time_w3", shape=[self._num_units], dtype=dtype)

        self._semantic_span_w1 = self.add_variable(
            "_semantic_span_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_span_b1 = self.add_variable(
            "_semantic_span_b1", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._semantic_span_w2 = self.add_variable(
            "_semantic_span_w2", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_span_b2 = self.add_variable(
            "_semantic_span_b2", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._semantic_last_w1 = self.add_variable(
            "_semantic_last_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_b1 = self.add_variable(
            "_semantic_last_b1", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._semantic_last_w2 = self.add_variable(
            "_semantic_last_w2", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_b2 = self.add_variable(
            "_semantic_last_b2", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._reconsume_span_w1 = self.add_variable(
            "_reconsume_span_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._reconsume_span_b1 = self.add_variable(
            "_reconsume_span_b1", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_last_w1 = self.add_variable(
            "_time_last_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_last_b1 = self.add_variable(
            "_time_last_b1", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._t_w11 = self.add_variable(
            "_t_w11", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._t_w12 = self.add_variable(
            "_t_w12", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._t_w13 = self.add_variable(
            "_t_w13", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._t_b11 = self.add_variable(
            "_t_b11", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)

        self.built = True

    def call(self, inputs, state):
        dtype = inputs.dtype
        is_reconsume = inputs[:, -1]
        time_last = inputs[:, -2]

        time_last_score = tf.expand_dims(time_last, axis=-1)
        reconsume_score = tf.expand_dims(is_reconsume, axis=-1)

        reconsume_emb = inputs[:,self._num_units: 2*self._num_units]

        inputs = inputs[:, :self._num_units]


        reconsume_state = state[:,self._num_units:]
        state = state[:, :self._num_units]


        # scope = variable_scope.get_variable_scope()
        # with variable_scope.variable_scope(scope) as unit_scope:
        #     with variable_scope.variable_scope(unit_scope):


        """
        v14
        """

        re_gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, reconsume_state], 1), self._reconsume_gate_kernel)
        re_gate_inputs = nn_ops.bias_add(re_gate_inputs, self._reconsume_gate_bias)
        re_value = math_ops.sigmoid(re_gate_inputs)

        re_r, re_u = array_ops.split(value=re_value, num_or_size_splits=2, axis=1)
        r_reconsume_state = re_r * reconsume_state

        re_candidate = math_ops.matmul(
            array_ops.concat([inputs, r_reconsume_state], 1), self._reconsume_candidate_kernel)
        re_candidate = nn_ops.bias_add(re_candidate, self._reconsume_candidate_bias)

        re_candidate = self._activation(re_candidate)

        # u*=reconsume_span_state

        new_reconsume_state = re_u * state + (1 - re_u) * re_candidate

        # temporary gate

        semantic_last_feature = tf.nn.relu(
            inputs * self._semantic_last_w1 + state * self._semantic_last_w2 + self._semantic_last_b1)
        semantic_span_feature = tf.nn.relu(
            inputs * self._semantic_span_w1 + state * self._semantic_span_w2 + self._semantic_span_b1)

        time_last_feature = tf.nn.relu(time_last_score * self._time_last_w1 + self._time_last_b1)
        reconsume_span_feature = new_reconsume_state

        # 计算 time aware gate
        time_last_state = tf.sigmoid(
            self._time_w1 * semantic_last_feature +
            self._time_w2 * time_last_feature +
            self._time_b1)
        reconsume_span_state = tf.sigmoid(
            self._t_w11 * semantic_span_feature +
            self._t_w12 * reconsume_span_feature +
            self._t_b11)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        value = math_ops.sigmoid(gate_inputs)

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        candidate = self._activation(candidate)

        # u*=reconsume_span_state


        new_h = u * state + (1 - u) * candidate

        new_h = array_ops.concat([new_h, new_reconsume_state ], 1)

        return new_h, new_h


class OnlyTimePredictionGRUCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(OnlyTimePredictionGRUCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return   self._num_units + 1

    @property
    def output_size(self):
        return   self._num_units + 1
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value -2
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_gate_kernel = self.add_variable(
            "_time_gate_kernel",
            shape=[3 * self._num_units, 2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_gate_bias = self.add_variable(
            "_time_gate_bias",
            shape=[2 * self._num_units],
            initializer=self._kernel_initializer)
        self._time_candidate_kernel = self.add_variable(
            "_time_candidate_kernel",
            shape=[3 * self._num_units, self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_candidate_bias = self.add_variable(
            "_time_candidate_bias",
            shape=[self._num_units],
            initializer=self._kernel_initializer)
    def call(self, inputs, state):
        dtype = inputs.dtype
        avg_interval = inputs[:, -1]
        time_last = inputs[:, -2]
        time_last_score = tf.expand_dims(time_last, axis=-1)

        time_emb = inputs[:, self._num_units:2 * self._num_units]
        inputs = inputs[:, :self._num_units]

        interval = state[:, -1]
        time_state = state[:, :self._num_units]

        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):

                self._time_pred_w = variable_scope.get_variable(
                    "_time_pred_w", shape=[self._num_units], dtype=dtype)
                self._time_pred_w2 = variable_scope.get_variable(
                    "_time_pred_w2", shape=[self._num_units], dtype=dtype)
                self._time_pred_b = variable_scope.get_variable(
                    "_time_pred_b", shape=[1], dtype=dtype)

                self._time_w1 = variable_scope.get_variable(
                    "_time_w1", shape=[self._num_units], dtype=dtype)
                self._time_b1 = variable_scope.get_variable(
                    "_time_b1", shape=[self._num_units], dtype=dtype)
                self._time_w2 = variable_scope.get_variable(
                    "_time_w2", shape=[self._num_units], dtype=dtype)
                self._time_b2 = variable_scope.get_variable(
                    "_time_b2", shape=[self._num_units], dtype=dtype)

                self._time_w3 = variable_scope.get_variable(
                    "_time_w3", shape=[self._num_units], dtype=dtype)
                self._time_b3 = variable_scope.get_variable(
                    "_time_b3", shape=[self._num_units], dtype=dtype)
                self._time_w4 = variable_scope.get_variable(
                    "_time_w4", shape=[self._num_units], dtype=dtype)
                self._time_b4 = variable_scope.get_variable(
                    "_time_b4", shape=[self._num_units], dtype=dtype)
                self._time_w5 = variable_scope.get_variable(
                    "_time_w5", shape=[self._num_units], dtype=dtype)
                self._time_w6 = variable_scope.get_variable(
                    "_time_w6", shape=[self._num_units], dtype=dtype)
                self._time_b5 = variable_scope.get_variable(
                    "_time_b5", shape=[self._num_units], dtype=dtype)
                self._semantic_span_w1 = variable_scope.get_variable(
                    "_semantic_span_w1", shape=[self._num_units], dtype=dtype)
                self._semantic_span_b1 = variable_scope.get_variable(
                    "_semantic_span_b1", shape=[self._num_units], dtype=dtype)
                self._semantic_span_w2 = variable_scope.get_variable(
                    "_semantic_span_w2", shape=[self._num_units], dtype=dtype)
                self._semantic_span_b2 = variable_scope.get_variable(
                    "_semantic_span_b2", shape=[self._num_units], dtype=dtype)
                self._semantic_last_w1 = variable_scope.get_variable(
                    "_semantic_last_w1", shape=[self._num_units], dtype=dtype)
                self._semantic_last_b1 = variable_scope.get_variable(
                    "_semantic_last_b1", shape=[self._num_units], dtype=dtype)
                self._semantic_last_w2 = variable_scope.get_variable(
                    "_semantic_last_w2", shape=[self._num_units], dtype=dtype)
                self._semantic_last_b2 = variable_scope.get_variable(
                    "_semantic_last_b2", shape=[self._num_units], dtype=dtype)
                self._time_span_w1 = variable_scope.get_variable(
                    "_time_span_w1", shape=[self._num_units], dtype=dtype)
                self._time_span_b1 = variable_scope.get_variable(
                    "_time_span_b1", shape=[self._num_units], dtype=dtype)
                self._time_last_w1 = variable_scope.get_variable(
                    "_time_last_w1", shape=[self._num_units], dtype=dtype)
                self._time_last_b1 = variable_scope.get_variable(
                    "_time_last_b1", shape=[self._num_units], dtype=dtype)
                self._t_w11 = variable_scope.get_variable(
                    "_t_w11", shape=[self._num_units], dtype=dtype)
                self._t_w12 = variable_scope.get_variable(
                    "_t_w12", shape=[self._num_units], dtype=dtype)
                self._t_w13 = variable_scope.get_variable(
                    "_t_w13", shape=[self._num_units], dtype=dtype)
                self._t_w14 = variable_scope.get_variable(
                    "_t_w14", shape=[self._num_units], dtype=dtype)
                self._t_w15 = variable_scope.get_variable(
                    "_t_w15", shape=[self._num_units], dtype=dtype)
                self._t_w16 = variable_scope.get_variable(
                    "_t_w16", shape=[self._num_units], dtype=dtype)
                self._t_b11 = variable_scope.get_variable(
                    "_t_b11", shape=[self._num_units], dtype=dtype)
                self._t_b12 = variable_scope.get_variable(
                    "_t_b12", shape=[self._num_units], dtype=dtype)
                self._t_b13 = variable_scope.get_variable(
                    "_t_b13", shape=[self._num_units], dtype=dtype)
                self._t_w1 = variable_scope.get_variable(
                    "_t_w1", shape=[1], dtype=dtype)
                self._t_b1 = variable_scope.get_variable(
                    "_t_b1", shape=[1], dtype=dtype)
                self._t_w2 = variable_scope.get_variable(
                    "_t_w2", shape=[1], dtype=dtype)
                self._t_b2 = variable_scope.get_variable(
                    "_t_b2", shape=[1], dtype=dtype)
                self._t_w3 = variable_scope.get_variable(
                    "_t_w3", shape=[1], dtype=dtype)
                self._t_b3 = variable_scope.get_variable(
                    "_t_b3", shape=[1], dtype=dtype)
                self._t_w4 = variable_scope.get_variable(
                    "_t_w4", shape=[1], dtype=dtype)
                self._t_b4 = variable_scope.get_variable(
                    "_t_b4", shape=[1], dtype=dtype)
                self._t_w5 = variable_scope.get_variable(
                    "_t_w5", shape=[1], dtype=dtype)
                self._t_w6 = variable_scope.get_variable(
                    "_t_w6", shape=[1], dtype=dtype)
                self._t_w7 = variable_scope.get_variable(
                    "_t_w7", shape=[1], dtype=dtype)
                self._t_w8 = variable_scope.get_variable(
                    "_t_w8", shape=[1], dtype=dtype)
                self._t_w9 = variable_scope.get_variable(
                    "_t_w9", shape=[1], dtype=dtype)
                self._t_w10 = variable_scope.get_variable(
                    "_t_w10", shape=[1], dtype=dtype)


        time_emb = tf.nn.relu(self._t_w11 * time_last_score + self._t_b11)  # batch_size, num_units

        time_gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, time_state, time_emb], 1), self._time_gate_kernel
        )
        time_gate_inputs = nn_ops.bias_add(time_gate_inputs, self._time_gate_bias)
        time_value = math_ops.sigmoid(time_gate_inputs)

        t_r, t_u = array_ops.split(value=time_value, num_or_size_splits=2, axis=1)
        time_reset_state = t_r * time_state

        time_candidate = math_ops.matmul(
            array_ops.concat([inputs,  time_state ,time_reset_state], 1), self._time_candidate_kernel
        )
        time_candidate = nn_ops.bias_add(time_candidate, self._time_candidate_bias)
        time_candidate = self._activation(time_candidate)

        new_time_state = t_u * time_state + (1 - t_u) * time_candidate

        new_time_value = tf.nn.relu(tf.reduce_mean(new_time_state * self._time_pred_w, axis=1) +
                                    tf.reduce_mean(inputs * self._time_pred_w2, axis=1) +
                                    self._t_w1 * time_last +
                                    self._time_pred_b)


        new_h = array_ops.concat([  new_time_state, tf.expand_dims(new_time_value, -1)], 1)

        return new_h, new_h


class TimePredictionGRUCell_v6(RNNCell):

    def __init__(self,num_units, forget_bias=1.0,activation=None,reuse=None):
        super(TimePredictionGRUCell_v6,self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return    self._num_units +1

    @property
    def output_size(self):
        return    self._num_units +1


    def call(self,inputs, state):
        dtype = inputs.dtype
        avg_interval = inputs[:, -1]
        time_last = inputs[:, -2]
        time_last_score = tf.expand_dims(time_last,axis=-1)

        inputs = inputs[:, :-2]

        interval = state[:, -1]  # 预测的时间
        state = state[:, :self._num_units]


        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                # weights for time now
                self._gate_kernel = variable_scope.get_variable(
                    "_gate_kernel", shape=[2*self._num_units,2*self._num_units], dtype=dtype)
                self._gate_bias = variable_scope.get_variable(
                    "_gate_bias", shape=[2 * self._num_units ], dtype=dtype)
                self._candidate_kernel = variable_scope.get_variable(
                    "_candidate_kernel", shape=[ 2 * self._num_units,  self._num_units], dtype=dtype)
                self._candidate_bias = variable_scope.get_variable(
                    "_candidate_bias", shape=[ self._num_units], dtype=dtype)
                self._time_kernel_w1 = variable_scope.get_variable(
                    "_time_kernel_w1", shape=[self._num_units], dtype=dtype)
                self._time_kernel_b1 = variable_scope.get_variable(
                    "_time_kernel_b1", shape=[self._num_units], dtype=dtype)
                self._time_history_w1 = variable_scope.get_variable(
                    "_time_history_w1", shape=[self._num_units], dtype=dtype)
                self._time_history_b1 = variable_scope.get_variable(
                    "_time_history_b1", shape=[self._num_units], dtype=dtype)
                self._time_w1 = variable_scope.get_variable(
                    "_time_w1", shape=[self._num_units], dtype=dtype)
                self._time_w2 = variable_scope.get_variable(
                    "_time_w2", shape=[self._num_units], dtype=dtype)
                self._time_w12 = variable_scope.get_variable(
                    "_time_w12", shape=[self._num_units], dtype=dtype)
                self._time_b1 = variable_scope.get_variable(
                    "_time_b1", shape=[self._num_units], dtype=dtype)
                self._time_b12 = variable_scope.get_variable(
                    "_time_b12", shape=[self._num_units], dtype=dtype)
                # weight for time last
                self._time_kernel_w2 = variable_scope.get_variable(
                    "_time_kernel_w2", shape=[self._num_units], dtype=dtype)
                self._time_kernel_b2 = variable_scope.get_variable(
                    "_time_kernel_b2", shape=[self._num_units], dtype=dtype)
                self._time_history_w2 = variable_scope.get_variable(
                    "_time_history_w2", shape=[self._num_units], dtype=dtype)
                self._time_history_b2 = variable_scope.get_variable(
                    "_time_history_b2", shape=[self._num_units], dtype=dtype)
                self._time_w2 = variable_scope.get_variable(
                    "_time_w2", shape=[self._num_units], dtype=dtype)
                self._time_b2 = variable_scope.get_variable(
                    "_time_b2", shape=[self._num_units], dtype=dtype)

                self._new_time_w1 = variable_scope.get_variable(
                    "_new_time_w1", shape=[self._num_units], dtype=dtype)
                self._new_time_b1 = variable_scope.get_variable(
                    "_new_time_b1", shape=[self._num_units], dtype=dtype)
                self._t_w1 = variable_scope.get_variable(
                    "_t_w1", shape=[1], dtype=dtype)
                self._t_w11 = variable_scope.get_variable(
                    "_t_w11", shape=[self._num_units], dtype=dtype)
                self._t_w12 = variable_scope.get_variable(
                    "_t_w12", shape=[self._num_units], dtype=dtype)
                self._t_w13 = variable_scope.get_variable(
                    "_t_w13", shape=[self._num_units], dtype=dtype)
                self._t_w14 = variable_scope.get_variable(
                    "_t_w14", shape=[self._num_units], dtype=dtype)
                self._t_w15 = variable_scope.get_variable(
                    "_t_w15", shape=[self._num_units], dtype=dtype)
                self._t_w16 = variable_scope.get_variable(
                    "_t_w16", shape=[self._num_units], dtype=dtype)
                self._t_b11 = variable_scope.get_variable(
                    "_t_b11", shape=[self._num_units], dtype=dtype)
                self._t_b1 = variable_scope.get_variable(
                    "_t_b1", shape=[1], dtype=dtype)
                self._t_w2 = variable_scope.get_variable(
                    "_t_w2", shape=[1], dtype=dtype)
                self._t_b2 = variable_scope.get_variable(
                    "_t_b2", shape=[1], dtype=dtype)
                self._t_w3 = variable_scope.get_variable(
                    "_t_w3", shape=[1], dtype=dtype)
                self._t_b3 = variable_scope.get_variable(
                    "_t_b3", shape=[1], dtype=dtype)
                self._t_w4 = variable_scope.get_variable(
                    "_t_w4", shape=[1], dtype=dtype)
                self._t_b4 = variable_scope.get_variable(
                    "_t_b4", shape=[1], dtype=dtype)
                self._t_w5 = variable_scope.get_variable(
                    "_t_w5", shape=[1], dtype=dtype)
                self._t_w6 = variable_scope.get_variable(
                    "_t_w6", shape=[1], dtype=dtype)
                self._t_w7 = variable_scope.get_variable(
                    "_t_w7", shape=[1], dtype=dtype)
                self._t_w8 = variable_scope.get_variable(
                    "_t_w8", shape=[1], dtype=dtype)
                self._t_w9 = variable_scope.get_variable(
                    "_t_w9", shape=[1], dtype=dtype)
                self._t_w10 = variable_scope.get_variable(
                    "_t_w10", shape=[1], dtype=dtype)
        """
        version 4
        """
        time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b1)  # batch_size, num_units

        new_t = nn_ops.relu(interval * self._t_w1 +
            tf.reduce_mean( time_feature * self._t_w14 + inputs * self._t_w15, 1) +self._t_b3)


        """
        version 5
        """
        time_feature = tf.nn.relu(self._t_w11 * time_last_score + self._t_b1)  # batch_size, num_units
        interval_feature = tf.nn.relu(self._t_w12 * tf.expand_dims(interval,axis=-1) + self._t_b2)

        new_t = nn_ops.relu( tf.reduce_mean(interval_feature * self._t_w13 + time_feature * self._t_w14 + inputs * self._t_w15, 1) +
                            self._t_b3)


        # time_last_score = tf.nn.relu(self._time_w1 * time_last_score + self._time_b1)
        # time_last_state = tf.sigmoid(
        #     self._time_w2 * time_feature + self._time_w12 * time_last_score + self._time_b12)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)

        # new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = (u * state + (1 - u) * c)

        new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        return new_h, new_h





class TimeGRUCell(GRUCell):
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
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = input_spec.InputSpec(ndim = 2)
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
            _check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1]-2
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
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self.output_w = self.add_variable(
                "output_w",
                shape=[self._num_units,self._num_units-1],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
    def call(self, inputs, state):
        dtype = inputs.dtype
        avg_interval = inputs[:, -1]
        time_last = inputs[:, -2]
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]

        interval = state[:,-1] # 预测的时间
        state = math_ops.matmul(state[:,:-1],self.output_w,transpose_b=True)
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


                self._new_time_w1 = variable_scope.get_variable(
                    "_new_time_w1", shape=[self._num_units], dtype=dtype)
                self._new_time_b1 = variable_scope.get_variable(
                    "_new_time_b1", shape=[self._num_units], dtype=dtype)
                self._t_w1 = variable_scope.get_variable(
                    "_t_w1", shape=[1], dtype=dtype)
                self._t_w11 = variable_scope.get_variable(
                    "_t_w11", shape=[self._num_units], dtype=dtype)
                self._t_w12 = variable_scope.get_variable(
                    "_t_w12", shape=[self._num_units], dtype=dtype)
                self._t_w13 = variable_scope.get_variable(
                    "_t_w13", shape=[self._num_units], dtype=dtype)
                self._t_b1 = variable_scope.get_variable(
                    "_t_b1", shape=[1], dtype=dtype)
                self._t_w2 = variable_scope.get_variable(
                    "_t_w2", shape=[1], dtype=dtype)
                self._t_b2 = variable_scope.get_variable(
                    "_t_b2", shape=[1], dtype=dtype)
                self._t_w3 = variable_scope.get_variable(
                    "_t_w3", shape=[1], dtype=dtype)
                self._t_b3 = variable_scope.get_variable(
                    "_t_b3", shape=[1], dtype=dtype)
                self._t_w4 = variable_scope.get_variable(
                    "_t_w4", shape=[1], dtype=dtype)
                self._t_b4 = variable_scope.get_variable(
                    "_t_b4", shape=[1], dtype=dtype)
                self._t_w5 = variable_scope.get_variable(
                    "_t_w5", shape=[1], dtype=dtype)
                self._t_w6 = variable_scope.get_variable(
                    "_t_w6", shape=[1], dtype=dtype)
                self._t_w7 = variable_scope.get_variable(
                    "_t_w7", shape=[1], dtype=dtype)
                self._t_w8 = variable_scope.get_variable(
                    "_t_w8", shape=[1], dtype=dtype)
                self._t_w9 = variable_scope.get_variable(
                    "_t_w9", shape=[1], dtype=dtype)
                self._t_w10 = variable_scope.get_variable(
                    "_t_w10", shape=[1], dtype=dtype)

        """
        以下是预测的下一个时间
        这个时间是对下一个行为预测的补充
        """

        # v2
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(
        #         inputs * self._t_w11 + state * self._t_w12) + time_last * self._t_w2 +  self._t_b2)
        # new_t = interval * self._t_w1 +candidate_interval * self._t_w3 + self._t_b3  # batch_size,

        # v1
        # r_t = math_ops.sigmoid(tf.reduce_mean(
        #     inputs * self._t_w11) + interval * self._t_w1 + time_last * self._t_w4 + avg_interval * self._t_w5 + self._t_b1)
        # z_t = math_ops.sigmoid(tf.reduce_mean(
        #     inputs * self._t_w12) + interval * self._t_w2 + time_last * self._t_w6 + avg_interval * self._t_w7 + self._t_b2)
        # candidate_interval = tf.nn.relu(r_t * interval + tf.reduce_mean(
        #     inputs * self._t_w3) + time_last * self._t_w8 + avg_interval * self._t_w9 + self._t_b3)
        # new_t = z_t * candidate_interval + (1 - z_t) * interval

        #v3

        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11) + time_last * self._t_w2 + interval * self._t_w1 + self._t_b2)
        # new_t = candidate_interval  # batch_size,

        #v4
        # b = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1+ self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + avg_interval * self._t_w3 + self._t_b2)
        # new_t = b * candidate_interval + (1 - b) * avg_interval  # batch_size,
        #
        #
        # new_reset_gate = tf.nn.sigmoid(self._new_time_w1*tf.expand_dims(new_t,-1)+self._new_time_b1)


        time_last_weight = tf.nn.relu(inputs * self._time_kernel_w1 + self._time_kernel_b1+state * self._time_history_w1)
        #version 2
        #time_last_score =  tf.nn.relu(self._time_w1 * tf.log(time_last_score + 1) + self._time_b1)
        time_last_score = tf.nn.relu(self._time_w1 * time_last_score+ self._time_b1)
        # 计算 time aware gate
        time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_last_score+self._time_b12)
        #time_last_score = tf.nn.relu(self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        # # TODO modified
        # r = r*new_reset_gate

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)

        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = math_ops.matmul((u * state + (1 - u) * c * time_last_state),self.output_w)

        """
        best result for yoochoose
        """
        # b = math_ops.sigmoid(time_last * self._t_w1+ tf.reduce_mean(inputs*self._t_w11+state*self._t_w12,1)+self._t_b1)
        # candidate_interval = tf.squeeze(nn_ops.relu(math_ops.matmul(inputs,self._t_w2)+self._t_b2))
        # new_t = b*nn_ops.relu(interval-time_last)+(1-b)*candidate_interval # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t,-1)], 1)
        """
        每个item后的interval服从指数分布，分布期望用网络求出来
        """
        # b = math_ops.sigmoid(
        #     time_last * self._t_w1 + tf.reduce_mean(inputs * self._t_w11 + state * self._t_w12, 1) + self._t_b1)
        # candidate_interval = tf.squeeze(nn_ops.relu(math_ops.matmul(inputs, self._t_w2) + self._t_b2))
        # new_t = b * interval + (1 - b) * candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        simplest
        """
        # b = math_ops.sigmoid(time_last * self._t_w1  + self._t_b1)
        # candidate_interval = tf.squeeze(nn_ops.relu(math_ops.matmul(inputs, self._t_w2) + self._t_b2))
        # new_t = b * interval + (1 - b) * candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 4
        """
        # candidate_interval = nn_ops.relu(tf.reduce_mean(inputs* self._t_w11) + time_last * self._t_w2 +self._t_b2)
        # new_t =   candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        """
        version 5
        """
        candidate_interval = nn_ops.relu(
            tf.reduce_mean(inputs * self._t_w11) + time_last * self._t_w2 + interval * self._t_w1 + self._t_b2)
        new_t = candidate_interval  # batch_size,
        new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 6
        """
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11 + state * self._t_w12) + time_last * self._t_w2 + interval * self._t_w1 + self._t_b2)
        # new_t = candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 7
        """
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(
        #         inputs * self._t_w11 + state * self._t_w12) + time_last * self._t_w2 +  self._t_b2)
        # new_t = interval * self._t_w1 +candidate_interval * self._t_w3 + self._t_b3  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 8
        """
        # b = math_ops.sigmoid(interval * self._t_w3 + time_last * self._t_w4 +   self._t_b4)
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11) + time_last * self._t_w2 +  self._t_b2)
        # new_t = b* interval + (1-b) * candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 9
        """
        # b = math_ops.sigmoid(time_last * self._t_w1 + tf.reduce_mean(inputs * self._t_w11 , 1) + self._t_b1)
        # candidate_interval = nn_ops.relu( time_last * self._t_w2 + self._t_b2)
        # new_t = b * interval + (1 - b) * candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 10
        """
        # b = math_ops.sigmoid(interval * self._t_w1 + tf.reduce_mean(inputs * self._t_w11, 1) + self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + self._t_b2)
        # new_t = b * interval + (1 - b) * candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        """
        version 11
        """
        # candidate_interval = nn_ops.relu(interval * self._t_w1 +  time_last * self._t_w2 + self._t_b2)
        # new_t = candidate_interval # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        """
        version 12
        """
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11) + tf.reduce_mean(new_h * self._t_w13) + time_last * self._t_w2 + interval * self._t_w1 + self._t_b2)
        # new_t = candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 13
        """
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11)  + time_last * self._t_w2 +  self._t_b2)
        # new_t = (candidate_interval + interval)/2 # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        """
        version 14
        """
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11) + time_last * self._t_w2 + self._t_b2)
        # new_t = candidate_interval   # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        """
        version 15
        """
        # candidate_interval = nn_ops.relu(
        #     tf.reduce_mean(inputs * self._t_w11) + time_last * self._t_w2 +
        #     interval * self._t_w1 + avg_interval * self._t_w3 + self._t_b2)
        # new_t = candidate_interval  # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        """
        version 16
        """
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + avg_interval * self._t_w3 + self._t_b2)
        # b = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) +  self._t_b1)
        # new_t = b * candidate_interval + (1-b)*avg_interval # batch_size,
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)


        """
        version 18
        """
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + self._t_b2)
        #
        # new_t = candidate_interval  # batch_size,
        #
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 19
        """
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + self._t_b2)
        #
        # new_t = self._t_w1 * interval + self._t_w2 * candidate_interval  # batch_size,
        #
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 17
        """
        # b = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1+ self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + avg_interval * self._t_w3 + self._t_b2)
        # new_t = b * candidate_interval + (1 - b) * avg_interval  # batch_size,
        #
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 20
        """
        # b = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1 + self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + avg_interval * self._t_w3 + self._t_b2)
        # new_t = b * candidate_interval + (1 - b) * interval  # batch_size,
        #
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        """
        version 21
        """
        # b = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1 + self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + avg_interval * self._t_w3 + self._t_b2)
        # new_t = b * candidate_interval + (1 - b) * time_last  # batch_size,

        """
        version 22
        """
        # r_t = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1 + self._t_b1)
        # z_t = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w12) + interval * self._t_w2 + self._t_b2)
        # candidate_interval = r_t * interval + tf.reduce_mean(inputs * self._t_w3) + self._t_b3
        # new_t = z_t * candidate_interval + (1-z_t) * interval
        """
        version 23
        """
        # r_t = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w11) + interval * self._t_w1 + time_last * self._t_w4 + avg_interval * self._t_w5 +self._t_b1)
        # z_t = math_ops.sigmoid(tf.reduce_mean(inputs * self._t_w12) + interval * self._t_w2 + time_last * self._t_w6 + avg_interval * self._t_w7 +self._t_b2)
        # candidate_interval = r_t * interval + tf.reduce_mean(inputs * self._t_w3) + time_last * self._t_w8 + avg_interval * self._t_w9 +self._t_b3
        # new_t = z_t * candidate_interval + (1 - z_t) * interval
        """
        version 24
        """
        # r_t = math_ops.sigmoid(tf.reduce_mean(
        #     inputs * self._t_w11) + interval * self._t_w1 + time_last * self._t_w4 + avg_interval * self._t_w5 + self._t_b1)
        # z_t = math_ops.sigmoid(tf.reduce_mean(
        #     inputs * self._t_w12) + interval * self._t_w2 + time_last * self._t_w6 + avg_interval * self._t_w7 + self._t_b2)
        # candidate_interval = tf.nn.relu(r_t * interval + tf.reduce_mean(
        #     inputs * self._t_w3) + time_last * self._t_w8 + avg_interval * self._t_w9 + self._t_b3)
        # new_t = z_t * candidate_interval + (1 - z_t) * interval


        new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)

        # b = math_ops.sigmoid(
        #     time_last * self._t_w1 + tf.reduce_mean(inputs * self._t_w11 + state * self._t_w12, 1) + self._t_b1)
        # candidate_interval = nn_ops.relu(time_last * self._t_w2 + self._t_b2)
        # new_t = b * interval + (1 - b) * candidate_interval
        # new_h = array_ops.concat([new_h, tf.expand_dims(new_t, -1)], 1)
        return new_h, new_h

class OriginTimeGRUCell(GRUCell):
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
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = input_spec.InputSpec(ndim = 2)
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
            _check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1]-2
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
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self.output_w = self.add_variable(
                "output_w",
                shape=[self._num_units,self._num_units-1],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
    def call(self, inputs, state):
        dtype = inputs.dtype
        time_now_score = tf.expand_dims(inputs[:, -1], -1)
        time_last = inputs[:, -2]
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]
        interval = state[:,-1] # 预测的时间
        state = math_ops.matmul(state[:,:-1],self.output_w,transpose_b=True)
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
                self._t_w1 = variable_scope.get_variable(
                    "_t_w1", shape=[1], dtype=dtype)
                self._t_w11 = variable_scope.get_variable(
                    "_t_w11", shape=[self._num_units], dtype=dtype)
                self._t_w12 = variable_scope.get_variable(
                    "_t_w12", shape=[self._num_units], dtype=dtype)
                self._t_b1 = variable_scope.get_variable(
                    "_t_b1", shape=[1], dtype=dtype)
                self._t_w2 = variable_scope.get_variable(
                    "_t_w2", shape=[1], dtype=dtype)
                self._t_b2 = variable_scope.get_variable(
                    "_t_b2", shape=[1], dtype=dtype)

        #time_now_weight = tf.nn.relu( inputs * self._time_kernel_w1+self._time_kernel_b1)
        time_last_weight = tf.nn.relu(inputs * self._time_kernel_w1 + self._time_kernel_b1+state * self._time_history_w1)
        #time_now_state = tf.sigmoid( time_now_weight+ self._time_w1*tf.log(time_now_score+1)+self._time_b12)

        #time_last_weight =  tf.nn.relu(inputs* self._time_kernel_w2+self._time_kernel_b2 +state * self._time_history_w2)
        #time_last_state = tf.sigmoid( time_last_weight+ self._time_w2*tf.log(time_last_score+1)+self._time_b2)

        #version 2
        #time_last_score =  tf.nn.relu(self._time_w1 * tf.log(time_last_score + 1) + self._time_b1)
        time_last_score = tf.nn.relu(self._time_w1 * time_last_score+ self._time_b1)
        time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_last_score+self._time_b12)
        #time_last_score = tf.nn.relu(self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)




        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #time_last_weight = tf.nn.relu(inputs * self._time_kernel_w2 + self._time_kernel_b2 + state * self._time_history_w2)
        #time_last_state = tf.sigmoid(time_last_weight + self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)
        #new_h = u * state *  time_now_state + (1 - u) * c * time_last_state 0.0185 0.0136
        #new_h = u * state + (1 - u) * c * time_now_state 0.0237 0.013
        #new_h = u * state * time_now_state + (1 - u) * c * time_last_state #no position 0.0211 0.0137
        #new_h = u * state + (1 - u) * c * time_now_state #no position 0.0211 0.0143
        #new_h = u * state + (1 - u) * c 0.0185 0.0138
        #####
        #sli_rec no position 0.026 0.0144
        #####
        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = math_ops.matmul((u * state + (1 - u) * c * time_last_state),self.output_w)


        b = math_ops.sigmoid(time_last * self._t_w1+ tf.reduce_mean(inputs*self._t_w11+state*self._t_w12,1)+self._t_b1)
        candidate_interval = nn_ops.relu(time_last*self._t_w2+self._t_b2)
        new_t = b*interval+(1-b)*candidate_interval
        new_h = array_ops.concat([new_h, tf.expand_dims(new_t,-1)], 1)
        return new_h, new_h



class TimeReconsumeGRUCell(GRUCell):
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
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = input_spec.InputSpec(ndim = 2)
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
            _check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1]-3 #TODO modified
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
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
            self.output_w = self.add_variable(
                "output_w",
                shape=[self._num_units,self._num_units-2],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
    def call(self, inputs, state):
        dtype = inputs.dtype
        time_now_score = tf.expand_dims(inputs[:, -2], -1)
        time_last = inputs[:, -3]
        time_last_score = tf.expand_dims(inputs[:, -3], -1)
        inputs = inputs[:, :-3]
        reconsume_last = inputs[:,-1]

        interval = state[:,-2] # 预测的时间
        is_reconsume = state[:,-1]
        state = math_ops.matmul(state[:,:-2],self.output_w,transpose_b=True) # 转为128
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
                self._t_w1 = variable_scope.get_variable(
                    "_t_w1", shape=[1], dtype=dtype)
                self._t_w11 = variable_scope.get_variable(
                    "_t_w11", shape=[self._num_units], dtype=dtype)
                self._t_w12 = variable_scope.get_variable(
                    "_t_w12", shape=[self._num_units], dtype=dtype)
                self._t_b1 = variable_scope.get_variable(
                    "_t_b1", shape=[1], dtype=dtype)
                self._t_w2 = variable_scope.get_variable(
                    "_t_w2", shape=[1], dtype=dtype)
                self._t_b2 = variable_scope.get_variable(
                    "_t_b2", shape=[1], dtype=dtype)

                self._r_w11 = variable_scope.get_variable(
                    "_r_w11", shape = [1], dtype=dtype)
                self._r_w12 = variable_scope.get_variable(
                    "_r_w12", shape=[self._num_units], dtype=dtype)
                self._r_w13 = variable_scope.get_variable(
                    "_r_w13", shape=[self._num_units], dtype=dtype)
                self._r_w14 = variable_scope.get_variable(
                    "_r_w14", shape=[1], dtype=dtype)
                self._r_b1 = variable_scope.get_variable(
                    "_r_b1", shape=[1], dtype=dtype)
                self._r_w2 = variable_scope.get_variable(
                    "_r_w2", shape=[1], dtype=dtype)
                self._r_b2 = variable_scope.get_variable(
                    "_r_b2", shape=[1], dtype=dtype)

        #time_now_weight = tf.nn.relu( inputs * self._time_kernel_w1+self._time_kernel_b1)
        time_last_weight = tf.nn.relu(inputs * self._time_kernel_w1 + self._time_kernel_b1+state * self._time_history_w1)
        #time_now_state = tf.sigmoid( time_now_weight+ self._time_w1*tf.log(time_now_score+1)+self._time_b12)

        #time_last_weight =  tf.nn.relu(inputs* self._time_kernel_w2+self._time_kernel_b2 +state * self._time_history_w2)
        #time_last_state = tf.sigmoid( time_last_weight+ self._time_w2*tf.log(time_last_score+1)+self._time_b2)

        #version 2
        #time_last_score =  tf.nn.relu(self._time_w1 * tf.log(time_last_score + 1) + self._time_b1)
        time_last_score = tf.nn.relu(self._time_w1 * time_last_score+ self._time_b1)
        time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_last_score+self._time_b12)
        #time_last_score = tf.nn.relu(self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)




        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #time_last_weight = tf.nn.relu(inputs * self._time_kernel_w2 + self._time_kernel_b2 + state * self._time_history_w2)
        #time_last_state = tf.sigmoid(time_last_weight + self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)
        #new_h = u * state *  time_now_state + (1 - u) * c * time_last_state 0.0185 0.0136
        #new_h = u * state + (1 - u) * c * time_now_state 0.0237 0.013
        #new_h = u * state * time_now_state + (1 - u) * c * time_last_state #no position 0.0211 0.0137
        #new_h = u * state + (1 - u) * c * time_now_state #no position 0.0211 0.0143
        #new_h = u * state + (1 - u) * c 0.0185 0.0138
        #####
        #sli_rec no position 0.026 0.0144
        #####
        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = math_ops.matmul((u * state + (1 - u) * c * time_last_state),self.output_w) # TODO 变成126位 为什么一定要这么操作一下


        b = math_ops.sigmoid(time_last * self._t_w1+ tf.reduce_mean(inputs*self._t_w11+state*self._t_w12,1)+self._t_b1)
        candidate_interval = nn_ops.relu(time_last*self._t_w2+self._t_b2)
        new_t = b*interval+(1-b)*candidate_interval
        new_h = array_ops.concat([new_h, tf.expand_dims(new_t,-1)], 1)

        e = math_ops.sigmoid(reconsume_last*self._r_w11 + tf.reduce_mean(inputs*self._r_w12+state*self._r_w13,1)+\
                             time_last* self._r_w14 + self._r_b1)
        candidate_reconsume = nn_ops.relu(reconsume_last*self._r_w2+self._r_b2)
        new_reconsume = e*is_reconsume + (1-e) * candidate_reconsume

        new_h = array_ops.concat([new_h,tf.expand_dims(new_reconsume,-1)],1)



        return new_h, new_h



