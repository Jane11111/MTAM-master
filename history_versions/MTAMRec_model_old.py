import tensorflow as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
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
        self.seq_length = self.embedding.get_embedding(self.num_units)
        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])

        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)
class Time_Aware_self_Attention_model(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(enc=self.behavior_list_embedding_dense,
                                                               num_units=self.num_units,
                                                               num_heads=self.num_heads, num_blocks=self.num_blocks,
                                                               dropout_rate=self.dropout_rate, is_training=True, reuse=False,
                                                               key_length=self.seq_length, query_length=self.seq_length,
                                                               t_querys=self.time_list, t_keys=self.time_list,
                                                               t_keys_length=self.max_len, t_querys_length=self.max_len
                                                               )
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.max_len,
                                                 width=self.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.predict_behavior_emb = long_term_prefernce
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output()
class Time_Aware_RNN_model(MTAMRec_model):

    def build_model(self):
        self.gru_net_ins = GRU()
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
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
class Time_Aware_Hybird_model(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(enc=self.behavior_list_embedding_dense,
                                                               num_units=self.num_units,
                                                               num_heads=self.num_heads, num_blocks=self.num_blocks,
                                                               dropout_rate=self.dropout_rate, is_training=True, reuse=False,
                                                               key_length=self.seq_length, query_length=self.seq_length,
                                                               t_querys=self.time_list, t_keys=self.time_list,
                                                               t_keys_length=self.max_len, t_querys_length=self.max_len
                                                               )
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
        self.output()

# Multi-hop Time-aware Attentive Memory network (MTAM)
class MTAM(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
        self.output()
class MTAM_no_time_aware_rnn(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
                                                                              input_data=self.behavior_list_embedding_dense,
                                                                              input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
        self.output()
class MTAM_no_time_aware_att(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
                                                                              input_data=self.behavior_list_embedding_dense,
                                                                              input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
        self.output()
class MTAM_via_rnn(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            user_history = self.short_term_intent_temp
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)

        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)

        #with tf.variable_scope('Switch'):
            #self.long_term_prefernce = hybird_preference
            #self.short_term_intent = self.short_term_intent

            #self.predict_behavior_emb = layer_norm(self.long_term_prefernce+self.short_term_intent)

        self.output()

class Time_Aware_Hybird_model_no_self_att_via_switch_network_hard(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            hybird_preference = layer_norm(hybird_preference)

        with tf.variable_scope('Switch'):
            self.long_term_prefernce = hybird_preference
            self.short_term_intent = self.short_term_intent

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent], 1)

            self.z = tf.layers.dense(inputs=self.z_concate, units=1,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regulation_rate))
            self.z = tf.round(tf.nn.sigmoid(self.z))

            self.predict_behavior_emb = self.z*self.long_term_prefernce+(1-self.z)*self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

        self.output()

class Time_Aware_Hybird_model_no_self_att_via_switch_network_hard_personalized(MTAMRec_model):

    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)
        with tf.variable_scope('StateEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                           tf.expand_dims(self.timelast_list, 2),
                                                           tf.expand_dims(self.timenow_list, 2)], 2)
            self.state_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length,-1),
                                                                              type='new')
            self.state = gather_indexes(batch_size=self.now_bacth_data_size,
                                                        seq_length=self.max_len,
                                                        width=self.num_units,
                                                        sequence_tensor=self.state_temp,
                                                        positions=self.mask_index - 1)
            self.state = layer_norm(self.state)




            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            hybird_preference = layer_norm(hybird_preference)

        with tf.variable_scope('Switch'):
            self.long_term_prefernce = hybird_preference
            self.short_term_intent = self.short_term_intent

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent, self.state], 1)
            self.z = tf.layers.dense(inputs=self.z_concate, units=1,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regulation_rate))
            self.z = tf.round(tf.nn.sigmoid(self.z))



            self.predict_behavior_emb = self.z*self.long_term_prefernce+(1-self.z)*self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

        self.output()

class Time_Aware_Hybird_model_no_self_att_via_switch_network_soft(MTAMRec_model):

    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):

            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)

            self.short_term_intent = layer_norm(self.short_term_intent)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)

        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            hybird_preference = layer_norm(hybird_preference)

        with tf.variable_scope('Switch'):
            self.long_term_prefernce = hybird_preference
            self.short_term_intent = self.short_term_intent

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent], 1)

            self.z = tf.layers.dense(inputs=self.z_concate, units=2,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regulation_rate))
            self.z = tf.nn.softmax(self.z)



            self.predict_behavior_emb = tf.multiply(tf.slice(self.z,[0,0],[-1,1]),self.long_term_prefernce)+\
                                            tf.multiply(tf.slice(self.z, [0, 1], [-1, 1]), self.short_term_intent)

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

        self.output()


class Time_Aware_Hybird_model_no_self_att_via_switch_network_soft_personalized(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)
        with tf.variable_scope('StateEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                           tf.expand_dims(self.timelast_list, 2),
                                                           tf.expand_dims(self.timenow_list, 2)], 2)
            self.state_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                  input_data=self.time_aware_gru_net_input,
                                                                  input_length=tf.add(self.seq_length,-1),
                                                                  type='new')
            self.state = gather_indexes(batch_size=self.now_bacth_data_size,
                                                        seq_length=self.max_len,
                                                        width=self.num_units,
                                                        sequence_tensor=self.state_temp,
                                                        positions=self.mask_index - 1)
            self.state = layer_norm(self.state)




            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)

        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            hybird_preference = layer_norm(hybird_preference)

        with tf.variable_scope('Switch'):
            self.long_term_prefernce = hybird_preference
            self.short_term_intent = self.short_term_intent

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent, self.state], 1)

            self.z = tf.layers.dense(inputs=self.z_concate, units=32,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regulation_rate),
                                     activation=tf.nn.relu)
            self.z = tf.layers.dense(inputs=self.z, units=1,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regulation_rate))
            self.z = tf.nn.sigmoid(self.z)

            self.predict_behavior_emb = self.z*self.long_term_prefernce+(1-self.z)*self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

        self.output()

class Time_Aware_Hybird_model_no_self_att_via_switch_network_add(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            hybird_preference = layer_norm(hybird_preference)

        with tf.variable_scope('Switch'):
            self.long_term_prefernce = hybird_preference
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.long_term_prefernce+self.short_term_intent)

        self.output()



class Time_Aware_Recommender_via_switch_network(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(enc=self.behavior_list_embedding_dense,
                                                               num_units=self.num_units,
                                                               num_heads=self.num_heads, num_blocks=self.num_blocks,
                                                               dropout_rate=self.dropout_rate, is_training=True, reuse=False,
                                                               key_length=self.seq_length, query_length=self.seq_length,
                                                               t_querys=self.time_list, t_keys=self.time_list,
                                                               t_keys_length=self.max_len, t_querys_length=self.max_len
                                                               )
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='time-aware')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = hybird_preference
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        with tf.variable_scope('OutputLayer'):
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.max_len,
                                                 width=self.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.long_term_prefernce = long_term_prefernce
            self.short_term_intent = self.short_term_intent
            self.hybird_preference = hybird_preference

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent, self.hybird_preference], 1)

            self.z = tf.layers.dense(inputs=self.z_concate, units=3,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regulation_rate))
            self.z = tf.nn.softmax(self.z)


            if self.FLAGS.PISTRec_type == 'hard':
                if tf.argmax(self.z) == 0:
                    self.predict_behavior_emb = self.long_term_prefernce
                elif tf.argmax(self.z) == 1:
                    self.predict_behavior_emb = self.short_term_intent
                else:
                    self.predict_behavior_emb = self.hybird_preference

            elif self.FLAGS.PISTRec_type == 'soft':
                self.predict_behavior_emb = tf.multiply(tf.slice(self.z,[0,0],[-1,1]),self.long_term_prefernce)+\
                                            tf.multiply(tf.slice(self.z, [0, 1], [-1, 1]), self.short_term_intent)+\
                                            tf.multiply(tf.slice(self.z, [0, 2], [-1, 1]), self.hybird_preference)
            elif self.FLAGS.PISTRec_type == 'short':
                self.predict_behavior_emb = self.short_term_intent
            elif self.FLAGS.PISTRec_type == 'long':
                self.predict_behavior_emb = self.long_term_prefernce
            elif self.FLAGS.PISTRec_type == 'hybird':
                self.predict_behavior_emb = self.hybird_preference

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

        self.output()












