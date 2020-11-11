# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 14:28
# @Author  : zxl
# @FileName: CLSTMRec.py

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model  import base_model
import numpy as np
from Model.Modules.continuous_time_rnn import ContinuousLSTM
from tensorflow.python.ops import math_ops
class CLSTM_model(base_model):

    def __init__(self, FLAGS,Embeding,sess):

        super(CLSTM_model, self).__init__(FLAGS, Embeding)  # 此处修改了
        self.now_bacth_data_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self.num_units = self.FLAGS.num_units
        self.num_heads = self.FLAGS.num_heads
        self.num_blocks = self.FLAGS.num_blocks
        self.dropout_rate = self.FLAGS.dropout
        self.regulation_rate = self.FLAGS.regulation_rate
        self.user_embedding, \
        self.behavior_list_embedding_dense, \
        self.item_list_emb, \
        self.category_list_emb, \
        self.position_list_emb, \
        self.time_list, \
        self.timelast_list, \
        self.timenow_list, \
        self.target, \
        self.seq_length,\
        self.reconsume_list, \
        self.is_reconsume, \
        self.item_list = self.embedding.get_embedding(self.num_units)
        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])
        # self.target_item_time = self.target[2]
        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)

class LSTM(CLSTM_model):
    def build_model(self):

        self.ctsm_model = ContinuousLSTM()

        with tf.variable_scope('ShortTermIntentEncoder',reuse=tf.AUTO_REUSE):


            time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense, ],
                                                  axis=2)


            self.short_term_intent_temp = self.ctsm_model.lstm_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_length,-1)) # batch_size, max_len, num_units
            emb = gather_indexes(batch_size=self.now_bacth_data_size,
                                           seq_length=self.max_len,
                                           width= self.num_units  ,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units

            self.predict_behavior_emb = layer_norm(emb )  # batch_size,  num_units

        self.output()

class MyLSTM(CLSTM_model):
    def build_model(self):

        self.ctsm_model = ContinuousLSTM()

        with tf.variable_scope('ShortTermIntentEncoder',reuse=tf.AUTO_REUSE):


            time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense, ],
                                                  axis=2)


            self.short_term_intent_temp = self.ctsm_model.my_lstm_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_length,-1)) # batch_size, max_len, num_units
            emb = gather_indexes(batch_size=self.now_bacth_data_size,
                                           seq_length=self.max_len,
                                           width= self.num_units  ,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units

            self.predict_behavior_emb = layer_norm(emb )  # batch_size,  num_units

        self.output()
class CLSTM(CLSTM_model):

    def get_state(self ):
        with tf.variable_scope('cstm_get_emb',reuse=tf.AUTO_REUSE):

            time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  tf.expand_dims(self.timelast_list, 2)],
                                                  axis=2)

            output = self.ctsm_model.ctsm_net(hidden_units=self.num_units,
                                           input_data = time_aware_gru_net_input,
                                           input_length= self.seq_length  -1) # TODO
            h_i_minus = gather_indexes(batch_size=self.now_bacth_data_size,
                                   seq_length=self.max_len,
                                   width=self.num_units ,
                                   sequence_tensor=output[:,:,-self.num_units:],
                                   positions=self.mask_index  -1)  # TODO 把上一个时刻的各种信息取出来
            state = gather_indexes(batch_size=self.now_bacth_data_size,
                                        seq_length=self.max_len,
                                        width=self.num_units * 4,
                                        sequence_tensor=output[:,:,:-self.num_units],
                                        positions=self.mask_index - 1)  # 把上一个时刻的各种信息取出来
            o_i,c_i,c_i_bar,delta_i   = array_ops.split(value=state,num_or_size_splits=4,axis=1) # batch_size, num_units
        return o_i,c_i,c_i_bar,delta_i ,h_i_minus


    def cal_ht (self,o_i,c_i,c_i_bar,delta_i, interval):

        interval = tf.reshape(interval,[-1,1])

        # g_i = g_i + delta_i * interval

        # c_ti = c_i_bar + (c_i-c_i_bar)* math_ops.exp(-delta_i * (time_last)) # batch_size,  num_units
        # c_t  = c_i_bar + c_i * g_i

        c_t = c_i_bar + (c_i - c_i_bar) * tf.exp(-delta_i * (interval)) # batch_size, num_units
        h_t = o_i * (2*tf.nn.sigmoid(2*c_t)-1) # batch_size, num_units

        return h_t

    def build_model(self):
        self.ctsm_model = ContinuousLSTM()

        last_time = tf.squeeze(gather_indexes(batch_size=self.now_bacth_data_size,
                                    seq_length=self.max_len,
                                    width=1,
                                    sequence_tensor=self.time_list,
                                    positions=self.mask_index - 1),axis=1)


        o_i, c_i, c_i_bar, delta_i,  h_i_minus = self.get_state()

        predict_target_lambda_emb = self.cal_ht(o_i, c_i, c_i_bar, delta_i, self.target[2]-last_time)

        self.predict_behavior_emb = predict_target_lambda_emb


        self.output()


class CGru(CLSTM_model):

    def get_state(self ):
        with tf.variable_scope('cgru_get_emb',reuse=tf.AUTO_REUSE):

            time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  tf.expand_dims(self.timelast_list, 2)],
                                                  axis=2)

            output = self.ctsm_model.cgru_net(hidden_units=self.num_units,
                                           input_data = time_aware_gru_net_input,
                                           input_length=tf.add(self.seq_length,-1))

            state = gather_indexes(batch_size=self.now_bacth_data_size,
                                        seq_length=self.max_len,
                                        width=self.num_units * 3,
                                        sequence_tensor=output,
                                        positions=self.mask_index - 1)  # 把上一个时刻的各种信息取出来
            h_i, h_i_bar, delta_i  = array_ops.split(value=state,num_or_size_splits=3,axis=1) # batch_size, num_units
        return h_i, h_i_bar, delta_i


    def cal_ht (self,h_i, h_i_bar, delta_i,interval):

        interval = tf.reshape(interval,[-1,1])

        h_t  = h_i_bar + (h_i - h_i_bar) * tf.exp(-delta_i * (interval))# batch_size,  num_units

        return h_t

    def build_model(self):
        self.ctsm_model = ContinuousLSTM()

        last_time = tf.squeeze(gather_indexes(batch_size=self.now_bacth_data_size,
                                    seq_length=self.max_len,
                                    width=1,
                                    sequence_tensor=self.time_list,
                                    positions=self.mask_index - 1),axis=1)


        h_i, h_i_bar, delta_i = self.get_state( )

        predict_target_lambda_emb = self.cal_ht(h_i, h_i_bar, delta_i,self.target[2]-last_time)

        self.predict_behavior_emb = predict_target_lambda_emb


        self.output()