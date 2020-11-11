# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 11:05
# @Author  : zxl
# @FileName: TMRRC_model_check.py


import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model_check import base_model_check
import numpy as np


class TMRRCRec_model(base_model_check):

    def __init__(self, FLAGS,Embeding,sess):

        super(TMRRCRec_model, self).__init__(FLAGS, Embeding)  # 此处修改了
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



class TMRRC_check(TMRRCRec_model):
    def build_model(self):
        print('--------------------num blocks-------------------------'+str(self.num_blocks))


        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense
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


            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_length,-1),  #
                                                                          type='new',
                                                                          scope='gru') # batch_size, max_len, num_units
            self.predict_all_time_interval = self.short_term_intent_temp[:,:,-1] # batch_size, max_len
            short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                           seq_length=self.max_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)

            # with tf.variable_scope('item_embedding'):
            #     self.short_term_intent_temp = self.gru_net_ins.origin_time_gru_net(hidden_units=self.num_units,
            #                                                                 input_data=time_aware_gru_net_input,
            #                                                                 input_length=tf.add(self.seq_length, -1),  #
            #                                                                 type='new',
            #                                                                 scope='gru')  # batch_size, max_len, num_units
            #     self.predict_all_time_interval = self.short_term_intent_temp[:, :, -1]  # batch_size, max_len
            #     short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
            #                                        seq_length=self.max_len,
            #                                        width=self.num_units,
            #                                        sequence_tensor=self.short_term_intent_temp,
            #                                        positions=self.mask_index - 1)  # batch_size, num_units
            #
            #     short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
            #
            # with tf.variable_scope('time_embedding'):
            #     self.time_intent_temp = self.gru_net_ins.origin_time_gru_net(hidden_units=self.num_units,
            #                                                                        input_data=time_aware_gru_net_input,
            #                                                                        input_length=tf.add(self.seq_length,
            #                                                                                            -1),  #
            #                                                                        type='new',
            #                                                                        scope='gru')  # batch_size, max_len, num_units
            #     time_intent = gather_indexes(batch_size=self.now_bacth_data_size,
            #                                        seq_length=self.max_len,
            #                                        width=self.num_units,
            #                                        sequence_tensor=self.time_intent_temp,
            #                                        positions=self.mask_index - 1)  # batch_size, num_units
            #
            #     self.interval_bar = tf.layers.dense(time_intent,units=1,activation=tf.nn.relu)


        with tf.variable_scope('NextItemDecoder',reuse=tf.AUTO_REUSE):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina,
                                                                       self.num_units,
                                                                       self.num_heads, self.num_blocks,
                                                                       self.dropout_rate, is_training=True,
                                                                       reuse=False,
                                                                       key_length=self.seq_length,
                                                                       query_length=tf.ones_like(
                                                                           short_term_intent4vallina[:, 0, 0],
                                                                           dtype=tf.int32),
                                                                       t_querys=tf.expand_dims(self.target[2], 1),
                                                                       t_keys=self.time_list,
                                                                       t_keys_length=self.max_len,
                                                                       t_querys_length=1)
            self.predict_behavior_emb = layer_norm(hybird_preference)  # batch_size,  num_units



        with tf.variable_scope('TimePredict',reuse=tf.AUTO_REUSE):
            """
            预测时间
            """

            # TODO need modified
            # self.predict_time = self.target[2]
            self.predict_is_reconsume = self.is_reconsume

            # self.last_time = gather_indexes(batch_size=self.now_bacth_data_size,
            #                                 seq_length=self.max_len,
            #                                 width=1,
            #                                 sequence_tensor=self.time_list,
            #                                 positions=self.mask_index - 1)  # 上一个时间
            #
            # self.pred_emb = self.gru_net_ins.gru_net(hidden_units=self.num_units,
            #                                                        input_data=tf.expand_dims(self.timelast_list, -1),
            #                                                        input_length=tf.add(self.seq_length, -1))
            # self.pred_emb = gather_indexes(batch_size=self.now_bacth_data_size,
            #                                         seq_length=self.max_len,
            #                                         width=self.num_units,
            #                                         sequence_tensor=self.short_term_intent_temp,
            #                                         positions=self.mask_index - 1)
            # self.interval = tf.layers.dense(self.pred_emb,units=1,activation=tf.nn.relu)
            # self.predict_time = tf.reshape(tf.reshape(self.last_time,[-1,])+tf.reshape(self.interval,[-1,]),[-1,])


            self.last_time = gather_indexes(batch_size=self.now_bacth_data_size,
                                            seq_length=self.max_len,
                                            width=1,
                                            sequence_tensor=self.time_list,
                                            positions=self.mask_index - 1)  # 上一个时间
            # self.last_interval = gather_indexes(batch_size = self.now_bacth_data_size,
            #                                    seq_length=self.max_len,
            #                                    width = 1,
            #                                    sequence_tensor=self.timelast_list,
            #                                    positions = self.mask_index)
            # self.sum_interval = tf.reduce_sum(self.timelast_list,axis=1,keepdims=True)# batch_size, 1
            # self.interval_bar = (self.sum_interval-self.last_interval)/tf.to_float(tf.reshape(self.seq_length-1,[-1,1]))
            # self.interval_bar = gather_indexes(batch_size=self.now_bacth_data_size,
            #                                    seq_length=self.max_len,
            #                                    width = 1,
            #                                    sequence_tensor=self.timelast_list,
            #                                    positions = self.mask_index-1)
            # self.interval = time_aware_attention.vanilla_attention4time(enc=user_history,
            #                                                              dec=short_term_intent4vallina,
            #                                                              num_units=self.num_units,
            #                                                              num_heads=self.num_heads,
            #                                                              num_blocks=self.num_blocks,
            #                                                              dropout_rate=self.dropout_rate,
            #                                                              is_training=True,
            #                                                              reuse=False,
            #                                                              key_length=self.seq_length - 1,  # TODO 需要减1
            #                                                              query_length=tf.ones_like(
            #                                                                  short_term_intent4vallina[:, 0, 0],
            #                                                                  dtype=tf.int32),
            #                                                              timelast_lst=self.timelast_list,
            #                                                              last_time=self.last_time,
            #                                                              time_interval= tf.reshape(self.interval_bar,[-1,1]) ,
            #                                                              t_keys=self.time_list,
            #                                                              t_keys_length=self.max_len,
            #                                                              t_querys_length=1)

            # masks = tf.sequence_mask(self.seq_length - 1, maxlen=self.max_len, dtype=tf.float32)
            # masked_timelast_lst = masks * self.timelast_list  # 把target time mask 掉 batch_size, max_seq_len
            # inputs = tf.concat([masked_timelast_lst,tf.reshape(self.interval_bar,[-1,1]),self.predict_behavior_emb],axis=1)
            # self.interval = tf.layers.dense(inputs,units=1)
            self.interval = self.interval_bar
            self.predict_time = tf.reshape(tf.reshape(self.last_time,[-1,])+tf.reshape(self.interval,[-1,]),[-1,])




        self.output()
