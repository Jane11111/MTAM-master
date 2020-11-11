# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 19:51
# @Author  : zxl
# @FileName: graph_neural_network.py

import math
import tensorflow.compat.v1 as tf
from Model.Modules.time_aware_rnn import TimeAwareGRUCell_for_gnn
from Model.Modules.graph_multihead_attention import GraphAttention
from Model.Modules.net_utils import gather_indexes,layer_norm


class modified_gated_GNN( ):

    def cosine(self,a, b):

        mul_val = tf.matmul(a, b, transpose_b=True)

        sqrt_a = tf.sqrt(tf.reduce_sum(a * a, axis=-1, keepdims=True))
        sqrt_b = tf.expand_dims(tf.sqrt(tf.reduce_sum(b * b, axis=-1)), axis=1)
        res = mul_val / (sqrt_a * sqrt_b + 1e-8)

        return res

    def generate_adj_emb(self, init_emb, now_batch_size, num_units, adj_in, adj_out):
        self.stdv = 1.0 / math.sqrt(num_units)

        self.W_in = tf.get_variable('W_in', shape=[num_units, num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [num_units, num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        fin_state = init_emb  # batch_size, max_len, num_units

        fin_state = tf.reshape(fin_state, [now_batch_size, -1, num_units])
        fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
                                            self.W_in) + self.b_in, [now_batch_size, -1, num_units])
        fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
                                             self.W_out) + self.b_out, [now_batch_size, -1, num_units])

        self.in_weight = tf.get_variable('in_weight', shape=[50, 50], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.out_weight = tf.get_variable('out_weight', shape=[50, 50], dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        pos_in_state = tf.matmul(self.in_weight , fin_state_in)
        pos_out_state = tf.matmul(self.out_weight , fin_state_out)



        # trans_pos_in_state = tf.layers.dense(pos_in_state, units = num_units,
        #                                      activation = tf.nn.relu,
        #                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
        #                                      bias_initializer=tf.constant_initializer(1),
        #                                      name='trans_pos_in_state' )
        # trans_pos_out_state = tf.layers.dense(pos_out_state, units=num_units,
        #                                      activation=tf.nn.relu,
        #                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
        #                                      bias_initializer=tf.constant_initializer(1),
        #                                      name='trans_pos_out_state' )
        # self.adj_in_weight = tf.get_variable('adj_in_weight', shape=[50, 50], dtype=tf.float32,
        #                                  initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        # self.adj_out_weight = tf.get_variable('adj_out_weight', shape=[50, 50], dtype=tf.float32,
        #                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        # self.adj_in_weight *=tf.to_float(tf.cast(adj_in, tf.bool))
        # self.adj_out_weight *=tf.to_float(tf.cast(adj_out,tf.bool))
        adj_in_state = tf.matmul( adj_in, fin_state_in)
        adj_out_state = tf.matmul(adj_out, fin_state_out)

        # trans_adj_in_state = tf.layers.dense(adj_in_state, units=num_units,
        #                                      activation=tf.nn.relu,
        #                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
        #                                      bias_initializer=tf.constant_initializer(1),
        #                                      name='trans_adj_in_state'
        #                                      )
        # trans_adj_out_state = tf.layers.dense(adj_out_state, units=num_units,
        #                                       activation=tf.nn.relu,
        #                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
        #                                       bias_initializer=tf.constant_initializer(1),
        #                                       name='trans_adj_out_state'
        #                                       )

        adj_vec_in = tf.layers.dense(tf.concat([fin_state_in,pos_in_state, adj_in_state], axis=-1), units=num_units,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
                                     bias_initializer=tf.constant_initializer(1),
                                     name='adj_vec_in'
                                     )
        # adj_vec_in = adj_vec_in +  trans_pos_in_state *  trans_adj_in_state
        # #
        adj_vec_out = tf.layers.dense(tf.concat([fin_state_out, pos_out_state, adj_out_state], axis=-1), units=num_units,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
                                      bias_initializer=tf.constant_initializer(1),
                                      name='adj_vec_in')
        # adj_vec_out = adj_vec_out + trans_pos_out_state * trans_adj_out_state

        # TODO
        # alpha_in = tf.nn.sigmoid(tf.get_variable('alpha_in', shape=[1,1,1], dtype=tf.float32,
        #                                  initializer=tf.random_uniform_initializer(0, 2*self.stdv)))
        # alpha_out = tf.nn.sigmoid(tf.get_variable('alpha_out', shape=[1,1,1], dtype=tf.float32,
        #                            initializer=tf.random_uniform_initializer(0, 2* self.stdv)))
        # adj_vec_in = alpha_in * pos_in_state +(1-alpha_in) * adj_in_state
        # adj_vec_out = alpha_out * pos_out_state + (1-alpha_out) * adj_out_state
        #
        # adj_vec_in = layer_norm(adj_vec_in)
        # adj_vec_out = layer_norm(adj_vec_out)

        # in_gate = tf.layers.dense(tf.concat([pos_in_state, adj_in_state], axis=-1), units=num_units,
        #                               activation=tf.nn.relu,
        #                               kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
        #                               bias_initializer=tf.constant_initializer(1),
        #                               name='in_gate') # batch_size, max_len, num_units
        # out_gate = tf.layers.dense(tf.concat([pos_out_state, adj_out_state], axis=-1), units=num_units,
        #                               activation=tf.nn.relu,
        #                               kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
        #                               bias_initializer=tf.constant_initializer(1),
        #                               name='out_gate')
        #
        # adj_vec_in = pos_in_state * in_gate + adj_in_state * (1-in_gate)
        # adj_vec_out = pos_out_state * out_gate + adj_out_state * (1-out_gate)

        # adj_vec_in = adj_in_state * pos_in_state
        # adj_vec_out = adj_out_state *pos_out_state

        # # TODO 如果只是加了自环呢？？？
        # adj_vec_in = adj_in_state
        # adj_vec_out = adj_out_state


        adj_vec = tf.concat([adj_vec_in, adj_vec_out], axis=-1)
        adj_vec = tf.reshape(adj_vec, [now_batch_size, -1, 2 * num_units])

        return adj_vec

    def generate_graph_emb(self, init_emb, now_batch_size, num_units, adj_in, adj_out,step=2):
        adj_in = adj_in
        adj_out = adj_out

        cell = tf.nn.rnn_cell.GRUCell(num_units)
        with tf.variable_scope('gru'):

            fin_state = init_emb
            for i in range(step):
                adj_vec = self.generate_adj_emb(fin_state, now_batch_size, num_units, adj_in, adj_out)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(adj_vec, [-1, 2 * num_units]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, num_units]))
                fin_state = tf.reshape(fin_state, [now_batch_size, -1, num_units])
            if step == 0:
                adj_vec = self.generate_adj_emb(fin_state, now_batch_size, num_units, adj_in, adj_out)
                fin_state = tf.layers.dense(adj_vec, units=num_units,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
                                      bias_initializer=tf.constant_initializer(1),
                                      name='projection')

        return fin_state
class gated_GNN( ):

    def generate_adj_emb(self,init_emb,now_batch_size,  num_units,adj_in,adj_out):
        self.stdv = 1.0 / math.sqrt(num_units)

        self.W_in = tf.get_variable('W_in', shape=[num_units, num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [num_units, num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        fin_state = init_emb # batch_size, max_len, num_units




        fin_state = tf.reshape(fin_state, [now_batch_size, -1, num_units])
        fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
                                            self.W_in) + self.b_in, [now_batch_size, -1, num_units])
        fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
                                             self.W_out) + self.b_out, [now_batch_size, -1, num_units])
        adj_vec = tf.concat([tf.matmul(adj_in, fin_state_in),
                        tf.matmul(adj_out, fin_state_out)], axis=-1)
        adj_vec = tf.reshape(adj_vec,[now_batch_size,-1,2*num_units])

        return adj_vec


    def generate_graph_emb(self,init_emb, now_batch_size,num_units,adj_in, adj_out,step=2):
        adj_in = adj_in
        adj_out = adj_out

        cell = tf.nn.rnn_cell.GRUCell( num_units)
        with tf.variable_scope('gru'):

            fin_state = init_emb
            for i in range( step):

                adj_vec = self.generate_adj_emb(fin_state,now_batch_size,num_units,adj_in,adj_out)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(adj_vec, [-1, 2 * num_units]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, num_units]))
                fin_state = tf.reshape(fin_state, [now_batch_size, -1, num_units])
            if step == 0:
                adj_vec = self.generate_adj_emb(fin_state, now_batch_size, num_units, adj_in, adj_out)
                fin_state = tf.layers.dense(adj_vec,units = num_units)

        return fin_state
    def generate_time_aware_emb(self,init_emb, adj_avg_time,now_batch_size,num_units,adj_in, adj_out,step=2):
        adj_in = adj_in
        adj_out = adj_out

        cell = TimeAwareGRUCell_for_gnn( num_units)
        with tf.variable_scope('time_aware_gru_gnn'):

            fin_state = init_emb
            for i in range( step):

                adj_vec = self.generate_adj_emb(fin_state, now_batch_size, num_units, adj_in, adj_out)
                inputs = tf.concat([tf.expand_dims(tf.reshape(adj_vec, [-1, 2 * num_units]), axis=1),
                                    tf.reshape(adj_avg_time, [-1,1 ,1] ),
                                    tf.reshape(adj_avg_time, [-1,1 ,1] ),],axis=2)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, inputs,
                                      initial_state=tf.reshape(fin_state, [-1, num_units]))
        return tf.reshape(fin_state, [now_batch_size, -1, num_units])


class OrderedGAT():

    def generate_graph_emb(self, init_emb, key_length,  num_units,
                           mask_adj, eid_emb ,num_head=1, step=2):
        graph_att_model = GraphAttention()
        emb = graph_att_model.ordered_gat_attention(enc = init_emb,
                                             key_length=key_length,
                                             mask_adj = mask_adj,
                                             eid_emb  = eid_emb ,
                                             num_units = num_units,
                                             num_heads = num_head,
                                             num_blocks = step,
                                             dropout_rate = 0.1,
                                             is_training = True,
                                             reuse = None
                                             )
        return layer_norm(emb)


class GAT():

    def generate_graph_emb(self, init_emb, key_length,  num_units ,
                           mask_adj,  num_head=1, step=2):
        graph_att_model = GraphAttention()
        emb = graph_att_model.gat_attention(enc = init_emb,
                                             key_length=key_length,
                                             mask_adj = mask_adj,
                                             num_units = num_units,
                                             num_heads = num_head,
                                             num_blocks = step,
                                             dropout_rate = 0.1,
                                             is_training = True,
                                             reuse = None
                                             )
        return layer_norm(emb)
class WGAT():

    def generate_graph_emb(self, init_emb, key_length,  num_units ,
                           mask_adj,  num_head=1, step=2,dropout_rate=0.1,
                           max_len = 50):
        graph_att_model = GraphAttention()
        emb = graph_att_model.weighted_gat_attention(enc = init_emb,
                                             key_length=key_length,
                                             mask_adj = mask_adj,
                                             num_units = num_units,
                                             num_heads = num_head,
                                             num_blocks = step,
                                             dropout_rate = dropout_rate,
                                             is_training = True,
                                             reuse = None,
                                             max_len = max_len
                                             )
        return layer_norm(emb)
class SelfAttention_GNN( ):


    def generate_graph_emb(self,init_emb, key_length, query_length, num_units, mask_adj,
                           num_head = 1, step=2,dropout_rate = 0.1):
        # TODO self attention + mask

        graph_att_model = GraphAttention()

        emb = graph_att_model.self_attention(enc = init_emb,
                                             key_length=key_length,
                                             query_length= query_length,
                                             mask_adj  = mask_adj ,
                                             num_units = num_units,
                                             num_heads = num_head,
                                             num_blocks = step,
                                             dropout_rate = dropout_rate,
                                             is_training = True,
                                             reuse = None
                                             )
        return layer_norm(emb)


class ordered_gated_GNN( ):

    def generate_adj_emb(self,init_emb,now_batch_size,  num_units,adj_in,adj_out,mask_adj_in, mask_adj_out,eid_emb_in,eid_emb_out):
        self.stdv = 1.0 / math.sqrt(num_units)
        edge_num_units = eid_emb_in.shape[-1]
        self.W_in = tf.get_variable('W_in', shape=[num_units, num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [num_units, num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.edge_W_in = tf.get_variable('edge_W_in', shape=[edge_num_units, num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.edge_b_in = tf.get_variable('edge_b_in', [num_units], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.edge_W_out = tf.get_variable('edge_W_out', [edge_num_units, num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.edge_b_out = tf.get_variable('edge_b_out', [num_units], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        fin_state = init_emb


        fin_state = tf.reshape(fin_state, [now_batch_size, -1, num_units])
        fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
                                            self.W_in) + self.b_in, [now_batch_size, -1, num_units])
        fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
                                             self.W_out) + self.b_out, [now_batch_size, -1, num_units])
        adj_vec_in = tf.matmul(adj_in, fin_state_in)
        adj_vec_out = tf.matmul(adj_out, fin_state_out)

        # 使用edge emb，计算的权重
        # TODO和item emb拼接一下？？？？
        # edge_adj_in = tf.reshape(tf.layers.dense(eid_emb_in,units = 1,activation= tf.nn.sigmoid),[now_batch_size,tf.shape(init_emb)[1],tf.shape(init_emb)[1]])
        # edge_adj_out = tf.reshape(tf.layers.dense(eid_emb_out, units = 1,activation=tf.nn.sigmoid),[now_batch_size,tf.shape(init_emb)[1],tf.shape(init_emb)[1]])
        #
        # edge_adj_in *= mask_adj_in
        # edge_adj_out *= mask_adj_out
        #
        # edge_fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
        #                                     self.edge_W_in) + self.edge_b_in, [now_batch_size, -1, num_units])
        # edge_fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, num_units]),
        #                                      self.edge_W_out) + self.edge_b_out, [now_batch_size, -1, num_units])
        # edge_vec_in = tf.matmul(edge_adj_in * adj_in, edge_fin_state_in) # TODO 要不要乘上adj_in
        # edge_vec_out = tf.matmul(edge_adj_out * adj_out, edge_fin_state_out)


        in_edge_info = tf.reduce_sum(tf.expand_dims(adj_in, axis=-1) * eid_emb_in,axis=2) # batch_size, max_len, edge_emb_size
        out_edge_info = tf.reduce_sum(tf.expand_dims(adj_out, axis=-1) * eid_emb_out,axis=2)
        edge_vec_in = tf.reshape(tf.matmul(tf.reshape(in_edge_info, [-1, edge_num_units]),
                                            self.edge_W_in) + self.edge_b_in, [now_batch_size, -1, num_units])
        edge_vec_out = tf.reshape(tf.matmul(tf.reshape(out_edge_info, [-1, edge_num_units]),
                                            self.edge_W_out) + self.edge_b_out, [now_batch_size, -1, num_units])
        adj_vec = tf.concat([adj_vec_in, adj_vec_out,edge_vec_in,edge_vec_out], axis=-1)
        adj_vec = tf.reshape(adj_vec,[now_batch_size,-1,4*num_units])

        return adj_vec


    def generate_graph_emb(self,init_emb, now_batch_size,num_units,adj_in, adj_out,mask_adj_in, mask_adj_out,eid_emb_in,eid_emb_out,step=2):
        adj_in = adj_in
        adj_out = adj_out

        cell = tf.nn.rnn_cell.GRUCell( num_units)
        with tf.variable_scope('gru'):

            fin_state = init_emb
            for i in range( step):

                adj_vec = self.generate_adj_emb(fin_state,now_batch_size,num_units,adj_in,adj_out,mask_adj_in, mask_adj_out,eid_emb_in,eid_emb_out)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(adj_vec, [-1, 4 * num_units]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, num_units]))
                fin_state = tf.reshape(fin_state, [now_batch_size, -1, num_units])
            if step == 0:
                adj_vec = self.generate_adj_emb(fin_state, now_batch_size, num_units, adj_in, adj_out,mask_adj_in, mask_adj_out,eid_emb_in, eid_emb_out)
                # fin_state = adj_vec
                fin_state = tf.layers.dense(adj_vec, units=num_units)
        return fin_state
