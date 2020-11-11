# -*- coding: utf-8 -*-
# @Time    : 2020/10/17 15:27
# @Author  : zxl
# @FileName: MSRTP.py


import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model_time import base_model_time
import numpy as np


class MSRTP_model(base_model_time):

    def __init__(self, FLAGS,Embeding,sess):

        super(MSRTP_model, self).__init__(FLAGS, Embeding)  # 此处修改了
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



class MSRTP(MSRTP_model):
    def build_model(self):
        print('--------------------num blocks-------------------------'+str(self.num_blocks))


        self.gru_net_ins = GRU()

        with tf.variable_scope('ShortTermIntentEncoder',reuse=tf.AUTO_REUSE):

            timefirst_lst = tf.reshape(self.time_list[:, 0], [-1, 1])

            idx = tf.range(start=1., limit=self.max_len , delta=1)
            idx0 = tf.constant([1.])
            idx = tf.concat([idx0, idx], axis=0)

            avg_interval_lst = (self.time_list - timefirst_lst) / idx

            time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  tf.expand_dims(self.timelast_list, 2),
                                                  tf.expand_dims(avg_interval_lst,2)],
                                                  axis=2)


            self.short_term_intent_temp = self.gru_net_ins.time_prediction_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_length,-1),  #
                                                                          type='new',
                                                                          scope='gru') # batch_size, max_len, num_units
            short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                           seq_length=self.max_len,
                                           width=2*self.num_units+1,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:self.num_units]# 前面是h
            # short_term_intent = tf.layers.dense(short_term_intent, self.num_units)

            self.predict_behavior_emb = layer_norm(short_term_intent)  # batch_size,  num_units

        with tf.variable_scope('TimePredict',reuse=tf.AUTO_REUSE):

            # TODO need modified


            self.last_time = gather_indexes(batch_size=self.now_bacth_data_size,
                                            seq_length=self.max_len,
                                            width=1,
                                            sequence_tensor=self.time_list,
                                            positions=self.mask_index - 1)  # 上一个时间

            self.interval = self.interval_bar
            self.predict_time = tf.reshape(tf.reshape(self.last_time,[-1,])+tf.reshape(self.interval,[-1,]),[-1,])

            # self.predict_time = self.target[2]


        self.output()
