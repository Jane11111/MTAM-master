# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 21:03
# @Author  : zxl
# @FileName: GRNNRec.py


import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.graph_recurrent_neural_network import GraphRNN
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
from Model.Modules.graph_neural_network import gated_GNN,SelfAttention_GNN,GAT,OrderedGAT
from tensorflow.python.ops import math_ops, init_ops, variable_scope, array_ops
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
        self.reconsume_list, \
        self.is_reconsume, \
        self.item_list = self.embedding.get_embedding(self.num_units)
        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])

        self.adj_in, self.adj_out, \
        self.adj_masks, self.adj_avg_time = self.embedding.get_adj_matrix()

        self.in_eid_embedding, self.out_eid_embedding, self.eid_embedding = self.embedding.get_edge_embedding()


        self.mask_adj_in = tf.to_float(tf.cast(self.adj_in, tf.bool))
        self.mask_adj_out = tf.to_float(tf.cast(self.adj_out, tf.bool))

        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)

class OrderedGatRnnRec(MTAMRec_model):

    def build_model(self):

        self.gru_net_ins = GraphRNN()
        self.gnn_model = OrderedGAT()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('graph_emb', reuse = tf.AUTO_REUSE):
            structure_emb = self.gnn_model.generate_graph_emb( init_emb=user_behavior_list_embedding,
                                                               key_length = self.seq_length,
                                                               num_units = self.num_units,
                                                               adj_in = self.adj_in,
                                                               adj_out = self.adj_out,
                                                               mask_adj_in = self.mask_adj_in,
                                                               mask_adj_out = self.mask_adj_out,
                                                               mask_adj = self.adj_masks,
                                                               eid_emb_in=self.in_eid_embedding,
                                                               eid_emb_out = self.out_eid_embedding,
                                                               num_head=1,
                                                               step=1) # batch_size, max_len, num_units * 2

        with tf.variable_scope('ShortTermIntentEncoder'):

            # in_emb, out_emb = array_ops.split(value=structure_emb, num_or_size_splits=2, axis=2)
            #
            # structure_emb = in_emb+out_emb
            structure_emb = tf.layers.dense(structure_emb,units = self.num_units)

            rnn_inputs = tf.concat([user_behavior_list_embedding,structure_emb],axis=2)

            self.short_term_intent_temp = self.gru_net_ins.simple_grnn_net(hidden_units=self.num_units,
                                                                   input_data=rnn_inputs,
                                                                   input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()

class GatRnnRec(MTAMRec_model):

    def build_model(self):

        self.gru_net_ins = GraphRNN()
        self.gnn_model = GAT()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('graph_emb', reuse = tf.AUTO_REUSE):
            structure_emb = self.gnn_model.generate_graph_emb( init_emb=user_behavior_list_embedding,
                                                               key_length = self.seq_length,
                                                               num_units = self.num_units,
                                                               adj_in = self.adj_in,
                                                               adj_out = self.adj_out,
                                                               mask_adj = self.adj_masks,
                                                               mask_adj_in = self.mask_adj_in,
                                                               mask_adj_out = self.mask_adj_out,
                                                               num_head=1,
                                                               step=1) # batch_size, max_len, num_units * 2

        with tf.variable_scope('ShortTermIntentEncoder'):

            # in_emb, out_emb = array_ops.split(value=structure_emb, num_or_size_splits=2, axis=2)
            #
            # structure_emb = in_emb+out_emb
            structure_emb = tf.layers.dense(structure_emb,units = self.num_units)

            rnn_inputs = tf.concat([user_behavior_list_embedding,structure_emb],axis=2)

            self.short_term_intent_temp = self.gru_net_ins.simple_grnn_net(hidden_units=self.num_units,
                                                                   input_data=rnn_inputs,
                                                                   input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()

class AttentiveGrnnRec(MTAMRec_model):

    def build_model(self):

        self.gru_net_ins = GraphRNN()
        self.gnn_model = SelfAttention_GNN()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('graph_emb', reuse = tf.AUTO_REUSE):
            structure_emb = self.gnn_model.generate_graph_emb( init_emb=user_behavior_list_embedding,
                                                               key_length = self.seq_length,
                                                               query_length = self.seq_length,
                                                               num_units = self.num_units,
                                                               adj_in = self.adj_in,
                                                               adj_out = self.adj_out,
                                                               mask_adj_in = self.mask_adj_in,
                                                               mask_adj_out = self.mask_adj_out,
                                                               num_head=1,
                                                               step=1) # batch_size, max_len, num_units * 2

        with tf.variable_scope('ShortTermIntentEncoder'):

            # in_emb, out_emb = array_ops.split(value=structure_emb, num_or_size_splits=2, axis=2)
            #
            # structure_emb = in_emb+out_emb
            structure_emb = tf.layers.dense(structure_emb,units = self.num_units)

            rnn_inputs = tf.concat([user_behavior_list_embedding,structure_emb],axis=2)

            self.short_term_intent_temp = self.gru_net_ins.simple_grnn_net(hidden_units=self.num_units,
                                                                   input_data=rnn_inputs,
                                                                   input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()


class GatedGrnnRec(MTAMRec_model):

    def build_model(self):

        self.gru_net_ins = GraphRNN()
        self.gated_gnn_model = gated_GNN()

        with tf.variable_scope('user_behavior_emb'):
            user_behavior_list_embedding = self.behavior_list_embedding_dense

        with tf.variable_scope('neighbor_emb',reuse=tf.AUTO_REUSE):
            structure_emb = self.gated_gnn_model.generate_adj_emb(init_emb=user_behavior_list_embedding,
                                                                  now_batch_size=self.now_bacth_data_size,
                                                                  num_units=self.num_units,
                                                                  adj_in=self.adj_in,
                                                                  adj_out=self.adj_out,

                                                                  ) # batch_size, max_len, num_units * 2

        with tf.variable_scope('ShortTermIntentEncoder'):

            # in_emb, out_emb = array_ops.split(value=structure_emb, num_or_size_splits=2, axis=2)
            #
            # structure_emb = in_emb+out_emb
            structure_emb = tf.layers.dense(structure_emb,units = self.num_units)

            grnn_inputs = tf.concat([user_behavior_list_embedding,structure_emb],axis=2)

            self.short_term_intent_temp = self.gru_net_ins.simple_grnn_net(hidden_units=self.num_units,
                                                                   input_data=grnn_inputs,
                                                                   input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()



