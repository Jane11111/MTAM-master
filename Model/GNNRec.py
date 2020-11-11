# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 20:24
# @Author  : zxl
# @FileName: GNNRec.py

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
from Model.Modules.graph_neural_network import gated_GNN

import numpy as np


class MTAMRec_model(base_model):

    def __init__(self, FLAGS,Embeding,sess):

        super(MTAMRec_model, self).__init__(FLAGS, Embeding)  # 此处修改了
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
        self.seq_length, \
        self.item_list = self.embedding.get_embedding(self.num_units)
        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])

        self.adj_in,self.adj_out = self.embedding.get_adj_matrix()
        self.adj_avg_time = self.embedding.get_adj_time() # batch_size, max_len

        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)


class TimeAwareSR_GNN(MTAMRec_model):

    def build_model(self):
        attention = Attention()
        self.gru_net_ins = GRU()
        self.ggnn_model = gated_GNN()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('ggnn_encoding'):
            self.short_term_intent_temp = self.ggnn_model.generate_time_aware_emb( init_emb=user_behavior_list_embedding,
                                                                              adj_avg_time = self.adj_avg_time,
                                                                              now_batch_size=self.now_bacth_data_size,
                                                                              num_units=self.num_units,
                                                                              adj_in=self.adj_in,
                                                                              adj_out=self.adj_out,
                                                                              step=self.FLAGS.graph_step)
            # with tf.variable_scope('ShortTermIntentEncoder'):
            #     self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
            #                                                            input_data=self.behavior_list_embedding_dense,
            #                                                            input_length=tf.add(self.seq_length, -1))
            user_history = self.short_term_intent_temp
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)

            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)

        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                            1, 1, self.dropout_rate, is_training=True,
                                                            reuse=False, key_length=self.seq_length,
                                                            query_length=tf.ones_like(
                                                                short_term_intent4vallina[:, 0, 0], dtype=tf.int32))
            self.predict_behavior_emb = tf.concat([self.short_term_intent, hybird_preference], 1)
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output_concat()


class GNN_T_Gru(MTAMRec_model):

    def build_model(self):
        self.gru_net_ins = GRU()
        self.ggnn_model = gated_GNN()
        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('ggnn_encoding'):
            gnn_emb = self.ggnn_model.generate_graph_emb(init_emb=user_behavior_list_embedding,
                                   now_batch_size=self.now_bacth_data_size,
                                   num_units=self.num_units,
                                   adj_in=self.adj_in,
                                   adj_out=self.adj_out,
                                   step=1)


        with tf.variable_scope('ShortTermIntentEncoder'):
            timenext_list = self.timelast_list[:,1:]
            zeros = tf.zeros(shape=(self.now_bacth_data_size,1))
            timenext_list = tf.concat([timenext_list,zeros],axis=1)
            self.time_aware_gru_net_input = tf.concat([gnn_emb,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(timenext_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()


class GNN_T_Att(MTAMRec_model):

    def build_model(self):
        self.gru_net_ins = GRU()
        self.ggnn_model = gated_GNN()
        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('ggnn_encoding'):
            gnn_emb = self.ggnn_model.generate_graph_emb(init_emb=user_behavior_list_embedding,
                                   now_batch_size=self.now_bacth_data_size,
                                   num_units=self.num_units,
                                   adj_in=self.adj_in,
                                   adj_out=self.adj_out,
                                   step=1)

        time_aware_attention = Time_Aware_Attention()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(gnn_emb,
                                                               self.num_units,
                                                               self.num_heads, self.num_blocks,
                                                               self.dropout_rate, is_training=True,
                                                               reuse=False, key_length=self.seq_length,
                                                               query_length=self.seq_length,
                                                               t_querys=self.time_list,
                                                               t_keys=self.time_list,
                                                               t_keys_length=self.max_len, t_querys_length=self.max_len)
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.predict_behavior_emb = long_term_prefernce
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output()