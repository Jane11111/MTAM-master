# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 22:36
# @Author  : zxl
# @FileName: MSRRP.py


import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model_reconsume import base_model_reconsume
import numpy as np


class MSRRP_model(base_model_reconsume):

    def __init__(self, FLAGS,Embeding,sess):

        super(MSRRP_model, self).__init__(FLAGS, Embeding)  # 此处修改了
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

        self.reconsume_lst_embedding,self.is_reconsume_embedding = \
            self.embedding.get_reconsume_embedding(tf.to_int32(self.reconsume_list),
                                                   tf.to_int32(self.is_reconsume))

        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])


        # self.target_item_time = self.target[2]
        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)





class MSRRP(MSRRP_model):
    def build_model(self):
        print('--------------------num blocks-------------------------'+str(self.num_blocks))


        self.gru_net_ins = GRU()

        with tf.variable_scope('ShortTermIntentEncoder',reuse=tf.AUTO_REUSE):


            time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  self.reconsume_lst_embedding,
                                                  tf.expand_dims(self.timelast_list, 2),
                                                  tf.expand_dims(self.reconsume_list,2)],
                                                  axis=2)


            self.short_term_intent_temp = self.gru_net_ins.reconsume_prediction_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_length,-1),  #
                                                                          type='new',
                                                                          scope='gru') # batch_size, max_len, num_units
            emb = gather_indexes(batch_size=self.now_bacth_data_size,
                                           seq_length=self.max_len,
                                           width= self.num_units * 2 ,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units
            # self.predict_is_reconsume = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = emb[:,:self.num_units]# 前面是h
            # short_term_intent = tf.layers.dense(short_term_intent, self.num_units)

            self.predict_behavior_emb = layer_norm(short_term_intent)  # batch_size,  num_units

            predict_reconsume_emb = emb[:,self.num_units:]
            reconsume_table = self.embedding.reconsume_emb_lookup_table
            reconsume_scores = tf.nn.softmax(tf.matmul(predict_reconsume_emb,reconsume_table,transpose_b=True))

            self.predict_is_reconsume = reconsume_scores[:,1]

            def cosine(q, a):
                pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
                pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
                pooled_mul_12 = tf.reduce_sum(q * a, 1)
                score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
                return score

            item_embs = tf.reshape(self.item_list_emb,[-1,self.num_units]) # batch_size * max_len, num_units

            predict_target_embs = tf.tile(self.predict_behavior_emb,[self.max_len,1]) # batch_size * max_len, num_units

            reconsume_scores = cosine(predict_target_embs,item_embs) # batch_size * max_len
            self.reconsume_scores = tf.reshape(reconsume_scores,[-1,self.max_len])






        self.output()
