# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 10:23
# @Author  : zxl
# @FileName: gnn_baseline_model.py

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
from Model.Modules.graph_neural_network import ordered_gated_GNN,gated_GNN,WGAT
import numpy as np


class GNNRec_model(base_model):

    def __init__(self, FLAGS,Embeding,sess):

        super(GNNRec_model, self).__init__(FLAGS, Embeding)  # 此处修改了
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

        self.adj_in, self.adj_out  = self.embedding.get_adj_matrix()
        self.mask_adj_in = tf.to_float(tf.cast(self.adj_in, tf.bool))
        self.mask_adj_out = tf.to_float(tf.cast(self.adj_out, tf.bool))
        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)
class OrderedSR_GNN(GNNRec_model):


    def build_model(self):
        attention = Attention()
        self.gru_net_ins = GRU()
        self.ggnn_model = ordered_gated_GNN()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('ggnn_encoding',reuse=tf.AUTO_REUSE):
            self.short_term_intent_temp = self.ggnn_model.generate_graph_emb(init_emb=user_behavior_list_embedding,
                                   now_batch_size=self.now_bacth_data_size,
                                   num_units=self.num_units,
                                   adj_in=self.adj_in,
                                   adj_out=self.adj_out,
                                   mask_adj_in=self.mask_adj_in,
                                   mask_adj_out=self.mask_adj_out,
                                   eid_emb_in=self.in_eid_embedding,
                                   eid_emb_out=self.out_eid_embedding,
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

class GC_SAN(GNNRec_model):


    def build_model(self):
        attention = Attention()

        self.ggnn_model = gated_GNN()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('ggnn_encoding',reuse=tf.AUTO_REUSE):
            self.gnn_emb_vec = self.ggnn_model.generate_graph_emb(init_emb=user_behavior_list_embedding,
                                   now_batch_size=self.now_bacth_data_size,
                                   num_units=self.num_units,
                                   adj_in=self.adj_in,
                                   adj_out=self.adj_out,
                                   step=self.FLAGS.graph_step)

            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.gnn_emb_vec,
                                                    positions=self.mask_index - 1) # batch_size, num_units


        with tf.variable_scope('self_attention',reuse=tf.AUTO_REUSE):
            self.att_emb_vec = attention.self_attention(enc = self.gnn_emb_vec,
                                                        num_units = self.num_units,
                                                        num_heads = self.num_heads,
                                                        num_blocks = self.num_blocks,
                                                        dropout_rate = self.dropout_rate,
                                                        is_training = True,
                                                        reuse = None,
                                                        key_length= self.seq_length,
                                                        query_length = self.seq_length)
            self.long_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                seq_length=self.max_len,
                                                width=self.num_units,
                                                sequence_tensor=self.att_emb_vec,
                                                positions=self.mask_index  ) # batch_size, num_units
        with tf.variable_scope('sess_emb', reuse=tf.AUTO_REUSE):
            eps = tf.get_variable('eps',[1],dtype=tf.float32)

            self.predict_behavior_emb = eps*self.short_term_intent +(1-eps) * self.long_term_intent

        self.output()

class SR_GNN(GNNRec_model):


    def build_model(self):
        attention = Attention()
        self.gru_net_ins = GRU()
        self.ggnn_model = gated_GNN()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('ggnn_encoding',reuse=tf.AUTO_REUSE):
            self.short_term_intent_temp = self.ggnn_model.generate_graph_emb(init_emb=user_behavior_list_embedding,
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

        with tf.variable_scope('session_emb'):
            hybird_preference = attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                            1, 1, self.dropout_rate, is_training=True,
                                                            reuse=False, key_length=self.seq_length,#TODO 是否需要减1
                                                            query_length=tf.ones_like(
                                                                short_term_intent4vallina[:, 0, 0], dtype=tf.int32))
            self.predict_behavior_emb = tf.concat([self.short_term_intent, hybird_preference], 1)
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output_concat()


class FGNN(GNNRec_model):

    def gru_set2set(self,init_emb, num_units,step):

        cell = tf.nn.rnn_cell.GRUCell(num_units)
        q_star = tf.zeros(shape=(self.now_bacth_data_size,num_units*2))
        for i in range(step):
            if i == 0:
                q, h = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(q_star, [-1, 2 * num_units]), axis=1) ,
                                      dtype=tf.float32)
            else:
                q, h = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(q_star, [-1, 2 * num_units]), axis=1),
                                      initial_state=tf.reshape(h, [-1, num_units]))
            q = tf.reshape(q, [-1, num_units])
            r = self.attention.vanilla_attention(dec=tf.expand_dims(q,axis=1),enc=init_emb,num_units=num_units,
                                                 num_heads=1,num_blocks=1,dropout_rate=self.dropout_rate,
                                                 is_training=True,reuse=None,
                                                 key_length = self.seq_length,# TODO 是否需要减1
                                                 query_length = tf.ones_like(self.seq_length, dtype=tf.int32))
            r = tf.reshape(r, [-1,num_units])
            q_star = tf.concat([q,r],axis=-1)
        return q_star


    def build_model(self):
        self.gru = GRU()
        self.attention = Attention()
        self.wgat_model = WGAT()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('WGAT_gnn_encoding',reuse=tf.AUTO_REUSE):
            self.gnn_emb = self.wgat_model.generate_graph_emb(init_emb = user_behavior_list_embedding,
                                                              key_length =self.seq_length,
                                                              num_units = self.num_units ,
                                                              mask_adj=self.adj_in,
                                                              num_head=self.FLAGS.graph_head,
                                                              step=self.FLAGS.graph_step,
                                                              dropout_rate=self.dropout_rate,
                                                              max_len = self.max_len)


        with tf.variable_scope('readout_function',reuse=tf.AUTO_REUSE):


            q_star = self.gru_set2set(init_emb=self.gnn_emb,
                                     num_units=self.num_units,
                                     step=self.FLAGS.FGNN_readout_step)
            hybrid_preference = tf.layers.dense(q_star, units = self.num_units)


            self.predict_behavior_emb = hybrid_preference
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output()



