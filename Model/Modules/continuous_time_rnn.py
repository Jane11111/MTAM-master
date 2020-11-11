# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 14:28
# @Author  : zxl
# @FileName: continuous_time_rnn.py
import logging
import tensorflow as tf
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



class ContinuousLSTM():


    def build_ctsm_cell(self,hidden_units):
        cell  = ContinuousLSTMCell(hidden_units)
        return MultiRNNCell([cell])

    def ctsm_net(self, hidden_units, input_data, input_length, scope='ctsm'):

        cell = self.build_ctsm_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs
    def build_cgru_cell(self,hidden_units):
        cell  = ContinuousGruCell(hidden_units)
        return MultiRNNCell([cell])

    def cgru_net(self, hidden_units, input_data, input_length, scope='cgru'):

        cell = self.build_cgru_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs
    def build_lstm_cell(self,hidden_units):
        cell  = LSTMCell(hidden_units)
        return MultiRNNCell([cell])

    def lstm_net(self, hidden_units, input_data, input_length, scope='lstm'):

        cell = self.build_lstm_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs
    def build_my_lstm_cell(self,hidden_units):
        cell  = MyLSTMCell(hidden_units)
        return MultiRNNCell([cell])

    def my_lstm_net(self, hidden_units, input_data, input_length, scope='lstm'):

        cell = self.build_my_lstm_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs
class ContinuousLSTMCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(ContinuousLSTMCell, self).__init__(
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
        return 5 * self._num_units

    @property
    def output_size(self):
        return 5 * self._num_units
    
    
    def build(self, inputs_shape):

        dtype = tf.float32
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value - 2 - self._num_units
        self.Wi = self.add_variable(
            "Wi", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer)  # input_size, num_units
        self.Ui = self.add_variable(
            "Ui", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer
        )  # num_units, num_units
        self.di = self.add_variable(
            "di", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wi_bar = self.add_variable(
            "Wi_bar", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer)  # input_size, num_units
        self.Ui_bar = self.add_variable(
            "Ui_bar", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer
        )  # num_units, num_units
        self.di_bar = self.add_variable(
            "di_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wf = self.add_variable(
            "Wf", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer)  # input_size, num_units
        self.Uf = self.add_variable(
            "Uf", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer
        )  # num_units, num_units
        self.df = self.add_variable(
            "df", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wf_bar = self.add_variable(
            "Wf_next_bar", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer)  # input_size, num_units
        self.Uf_bar = self.add_variable(
            "Uf_next_bar", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer
        )  # num_units, num_units
        self.df_bar = self.add_variable(
            "df_next_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wz = self.add_variable(
            "Wz", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer)  # input_size, num_units
        self.Uz = self.add_variable(
            "Uz", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer
        )  # num_units, num_units
        self.dz = self.add_variable(
            "dz", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wo = self.add_variable(
            "Wo", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer)  # input_size, num_units
        self.Uo = self.add_variable(
            "Uo", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer
        )  # num_units, num_units
        self.do = self.add_variable(
            "do", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wd = self.add_variable(
            "Wd", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer)  # input_size, num_units
        self.Ud = self.add_variable(
            "Ud", shape=[self._num_units, self._num_units], dtype=dtype,
             initializer=self._kernel_initializer
        )  # num_units, num_units
        self.dd = self.add_variable(
            "dd", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units

        self._semantic_last_w1 = self.add_variable(
            "_semantic_last_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_w2 = self.add_variable(
            "_semantic_last_w2", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_b1 = self.add_variable(
            "_semantic_last_b1", shape=[self._num_units], dtype=dtype,
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
        self._time_span_w1 = self.add_variable(
            "_time_span_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_span_b1 = self.add_variable(
            "_time_span_b1", shape=[self._num_units], dtype=dtype,
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
        self._semantic_last_w1_bar = self.add_variable(
            "_semantic_last_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_w2_bar = self.add_variable(
            "_semantic_last_w2_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_b1_bar = self.add_variable(
            "_semantic_last_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_last_w1_bar = self.add_variable(
            "_time_last_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_last_b1_bar = self.add_variable(
            "_time_last_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_span_w1_bar = self.add_variable(
            "_time_span_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_span_b1_bar = self.add_variable(
            "_time_span_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_w1_bar = self.add_variable(
            "_time_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_b1_bar = self.add_variable(
            "_time_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_w2_bar = self.add_variable(
            "_time_w2_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self.built = True

    def call(self,inputs, state):
        sigmoid = math_ops.sigmoid
        dtype = inputs.dtype

        k = inputs[:,:-1]
        time_last = tf.expand_dims(inputs[:,-1],axis=1)# 当前时间与上次时间的间隔

        inputs = inputs[:,:-1]
        input_size = inputs.get_shape().with_rank(2)[1]

        o_i,c_i,c_i_bar,delta_i,h_i = array_ops.split(value=state,num_or_size_splits=5,axis=1)


        c_ti = c_i_bar + (c_i-c_i_bar)* math_ops.exp(-delta_i * (time_last)) # batch_size,  num_units
        # c_ti = c_i_bar + c_i * g_i


        h_ti = o_i * (2* sigmoid(2* c_ti) - 1) # batch_size, num_units


        i_next = sigmoid(math_ops.matmul(k,self.Wi) + math_ops.matmul(h_ti, self.Ui) + self.di)
        f_next = sigmoid(math_ops.matmul(k,self.Wf) + math_ops.matmul(h_ti,self.Uf) + self.df)
        z_next = 2* sigmoid(math_ops.matmul(k,self.Wz) + math_ops.matmul(h_ti, self.Uz)+self.dz) - 1
        o_next = sigmoid(math_ops.matmul(k, self.Wo) + math_ops.matmul(h_ti, self.Uo) + self.do)

        i_next_bar = sigmoid(math_ops.matmul(k, self.Wi_bar) + math_ops.matmul(h_ti, self.Ui_bar) + self.di_bar)
        f_next_bar = sigmoid(math_ops.matmul(k, self.Wf_bar) + math_ops.matmul(h_ti, self.Uf_bar) + self.df_bar)

        # 计算 time aware gate

        semantic_last_feature_bar = tf.nn.relu(
            inputs * self._semantic_last_w1_bar + h_i * self._semantic_last_w2_bar + self._semantic_last_b1_bar)

        time_last_feature_bar = tf.nn.relu(time_last * self._time_last_w1_bar + self._time_last_b1_bar)


        g_i_bar = tf.sigmoid(
            self._time_w1_bar * semantic_last_feature_bar +
            self._time_w2_bar * time_last_feature_bar +
            self._time_b1_bar)

        semantic_last_feature = tf.nn.relu(
            inputs * self._semantic_last_w1 + h_ti * self._semantic_last_w2 + self._semantic_last_b1)

        time_last_feature = tf.nn.relu(time_last * self._time_last_w1 + self._time_last_b1)

        # 计算 time aware gate
        g_i = tf.sigmoid(
            self._time_w1 * semantic_last_feature +
            self._time_w2 * time_last_feature +
            self._time_b1)

        c_next = f_next * c_ti + i_next * z_next * g_i

        c_next_bar = f_next_bar * c_i_bar + i_next_bar * z_next * g_i_bar

        delta_next = tf.nn.softplus(math_ops.matmul(k,self.Wd) + math_ops.matmul(h_ti,self.Ud)+self.dd)



        next_state = array_ops.concat([o_next,c_next,c_next_bar,delta_next, h_ti],axis=1)

        return   next_state,next_state


class ContinuousGruCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(ContinuousGruCell, self).__init__(
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
        return 3 * self._num_units

    @property
    def output_size(self):
        return 3 * self._num_units

    def build(self, inputs_shape):

        dtype = tf.float32
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value - 2 - self._num_units
        self.Wu = self.add_variable(
            "Wu", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Uu = self.add_variable(
            "Uu", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.du = self.add_variable(
            "du", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wu_bar = self.add_variable(
            "Wu_bar", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Uu_bar = self.add_variable(
            "Uu_bar", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.du_bar = self.add_variable(
            "du_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wr = self.add_variable(
            "Wr", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Ur = self.add_variable(
            "Ur", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.dr = self.add_variable(
            "dr", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units

        self.Wz = self.add_variable(
            "Wz", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Uz = self.add_variable(
            "Uz", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.dz = self.add_variable(
            "dz", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units

        self.Wd = self.add_variable(
            "Wd", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Ud = self.add_variable(
            "Ud", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.dd = self.add_variable(
            "dd", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self._semantic_last_w1 = self.add_variable(
            "_semantic_last_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_w2 = self.add_variable(
            "_semantic_last_w2", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_b1 = self.add_variable(
            "_semantic_last_b1", shape=[self._num_units], dtype=dtype,
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
        self._time_span_w1 = self.add_variable(
            "_time_span_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_span_b1 = self.add_variable(
            "_time_span_b1", shape=[self._num_units], dtype=dtype,
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
        self._semantic_last_w1_bar = self.add_variable(
            "_semantic_last_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_w2_bar = self.add_variable(
            "_semantic_last_w2_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._semantic_last_b1_bar = self.add_variable(
            "_semantic_last_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_last_w1_bar = self.add_variable(
            "_time_last_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_last_b1_bar = self.add_variable(
            "_time_last_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_span_w1_bar = self.add_variable(
            "_time_span_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_span_b1_bar = self.add_variable(
            "_time_span_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_w1_bar = self.add_variable(
            "_time_w1_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
        self._time_b1_bar = self.add_variable(
            "_time_b1_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
        self._time_w2_bar = self.add_variable(
            "_time_w2_bar", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)

        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        dtype = inputs.dtype

        k = inputs[:, :-1]
        time_last = tf.expand_dims(inputs[:, -1], axis=1)  # 当前时间与上次时间的间隔

        inputs = inputs[:, :-1]
        input_size = inputs.get_shape().with_rank(2)[1]

        h_i, h_i_bar, delta_i = array_ops.split(value=state, num_or_size_splits=3, axis=1)


        h_ti = h_i_bar + (h_i - h_i_bar) * tf.exp(-delta_i * (time_last))# batch_size,  num_units



        r_next = sigmoid(math_ops.matmul(k, self.Wr) + math_ops.matmul(h_ti, self.Ur) + self.dr)
        u_next = sigmoid(math_ops.matmul(k, self.Wu) + math_ops.matmul(h_ti, self.Uu) + self.du)
        delta_next = tf.nn.softplus(math_ops.matmul(k, self.Wd) + math_ops.matmul(h_ti, self.Ud) + self.dd)

        candidate_h_next =  math_ops.tanh(math_ops.matmul(k, self.Wz) + math_ops.matmul(h_ti*r_next, self.Uz) +self.dz)

        u_next_bar = sigmoid(math_ops.matmul(k, self.Wu_bar) + math_ops.matmul(h_ti, self.Uu_bar) + self.du_bar)

        # 计算 time aware gate

        semantic_last_feature_bar = tf.nn.relu(
            inputs * self._semantic_last_w1_bar + h_i * self._semantic_last_w2_bar + self._semantic_last_b1_bar)

        time_last_feature_bar = tf.nn.relu(time_last * self._time_last_w1_bar + self._time_last_b1_bar)

        g_i_bar = tf.sigmoid(
            self._time_w1_bar * semantic_last_feature_bar +
            self._time_w2_bar * time_last_feature_bar +
            self._time_b1_bar)

        semantic_last_feature = tf.nn.relu(
            inputs * self._semantic_last_w1 + h_ti * self._semantic_last_w2 + self._semantic_last_b1)

        time_last_feature = tf.nn.relu(time_last * self._time_last_w1 + self._time_last_b1)

        # 计算 time aware gate
        g_i = tf.sigmoid(
            self._time_w1 * semantic_last_feature +
            self._time_w2 * time_last_feature +
            self._time_b1)


        h_next = u_next * h_ti + (1-u_next) * candidate_h_next
        h_next_bar = u_next_bar * h_i_bar + (1-u_next_bar) * candidate_h_next

        next_state = array_ops.concat([h_next, h_next_bar, delta_next], axis=1)

        return next_state, next_state


class MyLSTMCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(MyLSTMCell, self).__init__(
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
        return 2 * self._num_units

    @property
    def output_size(self):
        return   self._num_units

    def build(self, inputs_shape):

        dtype = tf.float32
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self.Wi = self.add_variable(
            "Wi", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Ui = self.add_variable(
            "Ui", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.di = self.add_variable(
            "di", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wi_bar = self.add_variable(
            "Wi_bar", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Ui_bar = self.add_variable(
            "Ui_bar", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.di_bar = self.add_variable(
            "di_bar", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wf = self.add_variable(
            "Wf", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Uf = self.add_variable(
            "Uf", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.df = self.add_variable(
            "df", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units

        self.Wz = self.add_variable(
            "Wz", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Uz = self.add_variable(
            "Uz", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.dz = self.add_variable(
            "dz", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )  # num_units
        self.Wo = self.add_variable(
            "Wo", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer)  # input_size, num_units
        self.Uo = self.add_variable(
            "Uo", shape=[self._num_units, self._num_units], dtype=dtype,
            initializer=self._kernel_initializer
        )  # num_units, num_units
        self.do = self.add_variable(
            "do", shape=[self._num_units], dtype=dtype,
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype))
        )

        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid

        k = inputs

        c_i_minus, h_i_minus  = array_ops.split(value=state, num_or_size_splits=2, axis=1)


        i_next = sigmoid(math_ops.matmul(k, self.Wi) + math_ops.matmul(h_i_minus, self.Ui) + self.di)
        f_next = sigmoid(math_ops.matmul(k, self.Wf) + math_ops.matmul(h_i_minus, self.Uf) + self.df)
        z_next = 2 * sigmoid(math_ops.matmul(k, self.Wz) + math_ops.matmul(h_i_minus, self.Uz) + self.dz) - 1
        o_next = sigmoid(math_ops.matmul(k, self.Wo) + math_ops.matmul(h_i_minus, self.Uo) + self.do)

        c_next = f_next * c_i_minus + i_next * z_next
        h_next = o_next * math_ops.tanh(c_next)


        next_state = array_ops.concat([ c_next, h_next], axis=1)

        return h_next, next_state