# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 13:53
# @Author  : zxl
# @FileName: graph_multihead_attention.py

import math
import tensorflow.compat.v1 as tf
conv1d = tf.layers.conv1d

class GraphAttention():

    def normalize(self, inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
          `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def feedforward(self, inputs,
                    num_units=[2048, 512],
                    scope="feedforward",
                    reuse=None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.normalize(outputs)

        return outputs

    def multihead_attention(self, queries,
                            keys,
                            key_length,
                            query_length,
                            mask_adj,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            scope="graph_multihead_attention",
                            reuse=None,
                            ):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          queries_length: A 1d tensor with shape of [N].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          keys_length:  A 1d tensor with shape of [N].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

          mask_adj: 有边为1  N, T_q, T_q
          adj: 度数邻接矩阵 N, T_q, T_q

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]
            self.stdv = 1.0 / math.sqrt(num_units)
            # Linear projections, C = # dim or column, T_x = # vectors or actions
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
                                bias_initializer=tf.constant_initializer(0 ),
                                )  # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu,
            #                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
            #                     bias_initializer=tf.constant_initializer(0 ),)  # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu,
            #                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
            #                     bias_initializer=tf.constant_initializer(0 ),)  # (N, T_k, C)
            K = keys
            V = keys

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            # query-key score matrix
            # each big score matrix is then split into h score matrix with same size
            # w.r.t. different part of the feature
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            # key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))# (N, T_k)
            key_masks = tf.sequence_mask(key_length, tf.shape(keys)[1])  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            #TODO  for graph
            graph_adj_masks = tf.tile(mask_adj, [num_heads,1, 1])
            key_masks = tf.to_float (key_masks) * graph_adj_masks
            key_masks = tf.cast(key_masks, tf.bool)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            # outputs = tf.where(tf.equal(key_masks, 0), outputs, paddings)  # (h*N, T_q, T_k)
            outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

            # Causality = Future blinding: No use, removed




            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            '''
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Attention vector
            att_vec = outputs

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)
            '''

            # Query Masking
            # query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.sequence_mask(query_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            print(outputs.shape.as_list())
            print(query_masks.shape.as_list())

            # Attention vector
            #########Tom Sun
            att_vec = outputs

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # TODO 经过全连接层，并且进行drop out

            outputs = tf.layers.dense(outputs, units=num_units,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=self.stdv),
                                bias_initializer=tf.constant_initializer(0 ),)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs, att_vec

    # encoder-decoder construct
    def self_attention(self, enc, mask_adj  , num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length,
                       query_length):
        with tf.variable_scope("encoder"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc, in_att_vec = self.multihead_attention(queries=enc,
                                                            keys=enc,
                                                            mask_adj = mask_adj ,
                                                            num_units=num_units,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate,
                                                            is_training=is_training,
                                                            scope="in_attention",
                                                            key_length=key_length,
                                                            query_length=query_length
                                                            )


                    ### Feed Forward
                    # enc = self.feedforward(enc,
                    # num_units=[num_units // 4, num_units],
                    # scope="feed_forward", reuse=reuse)

                    # self.self_attention_att_vec = stt_vec

            return enc
    def gat_attention(self, enc,  mask_adj,  num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length,
                      ):
        with tf.variable_scope("encoder"):
            for i in  range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):

                    enc, in_att_vec = self.gat_multihead_attention(seq=enc,
                                                            mask_adj = mask_adj ,
                                                            num_units=num_units,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate,
                                                            is_training=is_training,
                                                            scope="in_attention",
                                                            key_length=key_length,
                                                            )

            return enc

    def gat_multihead_attention(self, seq,
                            key_length,
                            mask_adj,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            scope="graph_multihead_attention",
                            reuse=None,
                            ):
        '''Applies multihead attention.

        Args:

          mask_adj: 有边为1  N, T_q, T_q
          adj: 度数邻接矩阵 N, T_q, T_q

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):

            res = tf.zeros_like(seq)

            for i in range(num_heads):
                with tf.variable_scope('head_'+str(i)):

                    seq_fts = tf.layers.conv1d(seq, num_units, 1, use_bias=False)  # batch_size, max_len, num_units

                    # simplest self-attention possible
                    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
                    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
                    outputs = f_1 + tf.transpose(f_2, [0, 2, 1])

                    outputs = tf.nn.leaky_relu(outputs)

                    # Key Masking

                    key_masks = tf.sequence_mask(key_length, tf.shape(seq)[1])  # (N, T_k)

                    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq)[1], 1])  # (N, T_q, T_k)

                    #TODO  for graph
                    graph_adj_masks = mask_adj # N, max_len, max_len
                    key_masks = tf.to_float (key_masks) * graph_adj_masks
                    key_masks = tf.cast(key_masks, tf.bool)
                    outputs *= graph_adj_masks


                    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
                    # outputs = tf.where(tf.equal(key_masks, 0), outputs, paddings)  # (h*N, T_q, T_k)
                    outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

                    # Causality = Future blinding: No use, removed


                    # Activation
                    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)


                    # Attention vector
                    att_vec = outputs

                    # Dropouts
                    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

                    # coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))  # TODO bias_in是什么

                    # if coef_drop != 0.0:
                    #     coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
                    # if in_drop != 0.0:
                    #     seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)


                    # TODO 这里是否需要激活函数
                    vals = tf.matmul(outputs, seq_fts) # batch_size, max_len, num_units

                    bias = tf.get_variable( "bias", shape=[num_units], dtype=tf.float32)
                    ret = tf.nn.bias_add(vals,bias)

                    # residual connection
                    ret = ret + seq
                    ret = tf.nn.elu(ret)

                    res+=ret
            res/=num_heads



        return res,att_vec

    def weighted_gat_attention(self, enc,  mask_adj,  num_units, num_heads, num_blocks,
                               dropout_rate, is_training, reuse, key_length, max_len = 50):
        with tf.variable_scope("encoder"):
            for i in  range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):

                    enc, in_att_vec = self.weighted_gat_multihead_attention(seq=enc,
                                                            mask_adj = mask_adj ,
                                                            num_units=num_units,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate,
                                                            is_training=is_training,
                                                            scope="in_attention",
                                                            key_length=key_length,
                                                            max_len = max_len
                                                            )

            return enc

    def weighted_gat_multihead_attention(self, seq,
                            key_length,
                            mask_adj,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            scope="weighted_graph_multihead_attention",
                            reuse=None,
                            max_len = 50
                            ):
        '''Applies multihead attention.

        Args:

          mask_adj: 有边为1  N, T_q, T_q
          adj: 度数邻接矩阵 N, T_q, T_q

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):

            res = tf.zeros_like(seq)
            # max_len = tf.shape(seq)[1]

            for i in range(num_heads):
                with tf.variable_scope('head_'+str(i)):

                    seq_fts = tf.layers.conv1d(seq, num_units, 1, use_bias=False)  # batch_size, max_len, num_units

                    # simplest self-attention possible
                    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
                    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
                    outputs = f_1 + tf.transpose(f_2, [0, 2, 1])

                    w_weight = tf.get_variable('w_weight', shape=[max_len, max_len], dtype=tf.float32  )
                    outputs += mask_adj*w_weight


                    outputs = tf.nn.leaky_relu(outputs)

                    # Key Masking

                    key_masks = tf.sequence_mask(key_length, tf.shape(seq)[1])  # (N, T_k)

                    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq)[1], 1])  # (N, T_q, T_k)

                    #TODO  for graph
                    graph_adj_masks = mask_adj # N, max_len, max_len
                    key_masks = tf.to_float (key_masks) * graph_adj_masks
                    key_masks = tf.cast(key_masks, tf.bool)


                    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
                    # outputs = tf.where(tf.equal(key_masks, 0), outputs, paddings)  # (h*N, T_q, T_k)
                    outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

                    # Causality = Future blinding: No use, removed


                    # Activation
                    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)


                    # Attention vector
                    att_vec = outputs

                    # Dropouts
                    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

                    # coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))  # TODO bias_in是什么

                    # if coef_drop != 0.0:
                    #     coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
                    # if in_drop != 0.0:
                    #     seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)


                    # TODO 这里是否需要激活函数
                    vals = tf.matmul(outputs, seq_fts) # batch_size, max_len, num_units

                    bias = tf.get_variable( "bias", shape=[num_units], dtype=tf.float32)
                    ret = tf.nn.bias_add(vals,bias)

                    # residual connection
                    ret = ret + seq
                    ret = tf.nn.relu(ret)

                    res+=ret
            res = tf.nn.relu(res/num_heads)


        return res,att_vec
    def ordered_gat_attention(self, enc,  mask_adj, eid_emb ,num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length,
                      ):
        with tf.variable_scope("encoder"):
            for i in  range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention

                    enc, out_att_vec = self.ordered_gat_multihead_attention(seq=enc,
                                                               mask_adj=mask_adj,
                                                               eid_emb_adj = eid_emb ,
                                                               num_units=num_units,
                                                               num_heads=num_heads,
                                                               dropout_rate=dropout_rate,
                                                               is_training=is_training,
                                                               scope="out_attention",
                                                               key_length=key_length,
                                                               )

                    ### Feed Forward
                    # enc = self.feedforward(enc,
                    # num_units=[num_units // 4, num_units],
                    # scope="feed_forward", reuse=reuse)

                    # self.self_attention_att_vec = stt_vec

            return enc

    def ordered_gat_multihead_attention(self, seq,
                            key_length,
                            mask_adj,
                            eid_emb_adj,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            scope="graph_multihead_attention",
                            reuse=None,
                            ):
        '''Applies multihead attention.

        Args:

          mask_adj: 有边为1  N, T_q, T_q
          adj: 度数邻接矩阵 N, T_q, T_q

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):

            res = tf.zeros_like(seq)

            for i in range(num_heads):
                with tf.variable_scope('head_'+str(i)):

                    seq_fts = tf.layers.conv1d(seq, num_units, 1, use_bias=False)  # batch_size, max_len, num_units

                    # simplest self-attention possible
                    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
                    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
                    outputs = f_1 + tf.transpose(f_2, [0, 2, 1])

                    # TODO 增加边权重
                    eid_att_vec = tf.layers.dense(eid_emb_adj,units=1)
                    eid_att_vec = tf.reshape(eid_att_vec,[-1,tf.shape(seq)[1],tf.shape(seq)[1]])
                    outputs += eid_att_vec

                    outputs = tf.nn.leaky_relu(outputs)

                    # Key Masking

                    key_masks = tf.sequence_mask(key_length, tf.shape(seq)[1])  # (N, T_k)

                    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq)[1], 1])  # (N, T_q, T_k)

                    #TODO  for graph
                    graph_adj_masks = mask_adj # N, max_len, max_len
                    key_masks = tf.to_float (key_masks) * graph_adj_masks
                    key_masks = tf.cast(key_masks, tf.bool)
                    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
                    # outputs = tf.where(tf.equal(key_masks, 0), outputs, paddings)  # (h*N, T_q, T_k)
                    outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

                    # Causality = Future blinding: No use, removed


                    # Activation
                    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)


                    # Attention vector
                    att_vec = outputs

                    # Dropouts
                    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

                    # coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))  # TODO bias_in是什么

                    # if coef_drop != 0.0:
                    #     coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
                    # if in_drop != 0.0:
                    #     seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)



                    vals = tf.matmul(outputs, seq_fts) # batch_size, max_len, num_units

                    bias = tf.get_variable( "bias", shape=[num_units], dtype=tf.float32)
                    ret = tf.nn.bias_add(vals,bias)

                    # residual connection
                    ret = ret + seq
                    ret = tf.nn.elu(ret)

                    res+=ret
            res/=num_heads



        return res,att_vec
    def vanilla_attention(self, enc, dec, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse,
                          key_length, query_length):

        # dec = tf.expand_dims(dec, 1)#在1的位置上增加1维
        with tf.variable_scope("decoder"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( vanilla attention)
                    dec, att_vec = self.multihead_attention(queries=dec,
                                                            keys=enc,
                                                            num_units=num_units,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate,
                                                            is_training=is_training,
                                                            scope="vanilla_attention",
                                                            key_length=key_length,
                                                            query_length=query_length)

                    ## Feed Forward
                    # dec = self.feedforward(dec,num_units=[num_units // 4, num_units],
                    # scope="feed_forward", reuse=reuse)

                    self.vanilla_attention_att_vec = att_vec

        # 此处怀疑有错误，非常重要
        dec = tf.reshape(dec, [-1, num_units])
        return dec



