# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 17:33
# @Author  : zxl
# @FileName: NHP.py


import tensorflow as tf
from Model.base_model import base_model
from Model.Modules.transformer_encoder import transformer_encoder
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.Modules.time_prediction import thp_time_predictor
from Model.Modules.type_prediction import thp_type_predictor
from Model.Modules.continuous_time_rnn import ContinuousLSTM
from tensorflow.python.ops import array_ops
from Model.Modules.intensity_calculation import nhp_intensity_calculation

class NHP_model(base_model):
    def __init__(self, FLAGS, Embedding, sess):

        super(NHP_model,self).__init__(FLAGS= FLAGS,
                                                Embedding= Embedding)
        self.now_batch_size = tf.placeholder(tf.int32, shape = [], name = 'bath_size')

        self.type_emb_size = self.FLAGS.type_emb_size
        self.num_units = self.FLAGS.num_units # attention的num
        self.num_heads = self.FLAGS.num_heads
        self.num_blocks = self.FLAGS.num_blocks
        self.dropout_rate = self.FLAGS.dropout
        self.regulation_rate = self.FLAGS.regulation_rate
        self.type_num = self.FLAGS.type_num
        self.sims_len = self.FLAGS.sims_len
        self.max_seq_len = self.FLAGS.max_seq_len

        self.type_lst, \
        self.type_lst_embedding, \
        self.time_lst, \
        self.time_lst_embedding, \
        self.sahp_time_lst_embedding, \
        self.target_type_embedding, \
        self.target_type,\
        self.target_time, \
        self.seq_len, \
        self.T_lst,\
        self.not_first_lst,\
        self.sims_time_lst,\
        self.target_time_last_lst,\
        self.target_time_now_lst, \
        self.sim_time_last_lst, \
        self.sim_time_now_lst = self.embedding.get_embedding(self.type_emb_size)
        self.mask_index = tf.reshape(self.seq_len - 1, [-1, 1])

        self.build_model()
        self.init_variables(sess)

class NHP(NHP_model):

    def get_state(self,time_last):
        with tf.variable_scope('cstm_get_emb',reuse=tf.AUTO_REUSE):
            ctsm_input = tf.concat([self.type_lst_embedding,
                                    tf.expand_dims(time_last, 2)],
                                    axis=2) # TODO 需要修改输入

            output = self.ctsm_model.ctsm_net(hidden_units=self.num_units,
                                           input_data = ctsm_input,
                                           input_length=tf.add(self.seq_len,-1))
            h_i_minus = gather_indexes(batch_size=self.now_batch_size,
                                   seq_length=self.max_seq_len,
                                   width=self.num_units ,
                                   sequence_tensor=output[:,:,-self.num_units:],
                                   positions=self.mask_index - 1)  # 把上一个时刻的各种信息取出来
            state = gather_indexes(batch_size=self.now_batch_size,
                                        seq_length=self.max_seq_len,
                                        width=self.num_units * 4,
                                        sequence_tensor=output[:,:,:-self.num_units],
                                        positions=self.mask_index - 1)  # 把上一个时刻的各种信息取出来
            o_i,c_i,c_i_bar,delta_i  = array_ops.split(value=state,num_or_size_splits=4,axis=1) # batch_size, num_units
        return o_i,c_i,c_i_bar,delta_i,h_i_minus


    def cal_ht (self,o_i,c_i,c_i_bar,delta_i,interval):

        interval = tf.reshape(interval,[-1,1])

        c_t = c_i_bar + (c_i - c_i_bar) * tf.exp(-delta_i * (interval)) # batch_size, num_units
        h_t = o_i * (2*tf.nn.sigmoid(2*c_t)-1) # batch_size, num_units

        return h_t

    def build_model(self):
        self.ctsm_model = ContinuousLSTM()
        self.transformer_model = transformer_encoder()

        last_time = tf.squeeze(gather_indexes(batch_size=self.now_batch_size,
                                    seq_length=self.max_seq_len,
                                    width=1,
                                    sequence_tensor=self.time_lst,
                                    positions=self.mask_index - 1),axis=1)


        o_i, c_i, c_i_bar, delta_i, h_i_minus = self.get_state(self.target_time_last_lst)

        predict_target_lambda_emb = self.cal_ht(o_i, c_i, c_i_bar, delta_i,self.target_time-last_time)

        # sims_time_lst: batch_size, sims_len
        predict_sims_emb = tf.zeros([self.now_batch_size, 1])
        self.test = tf.split(self.sims_time_lst, self.sims_len, 1)
        sims_time = tf.squeeze(tf.split(self.sims_time_lst, self.sims_len, 1), 2)

        for i in range(self.sims_len):
            # 第i个时间 batch_size, num_units

            cur_sims_emb= self.cal_ht(o_i, c_i, c_i_bar,delta_i,sims_time[i]-last_time)
            predict_sims_emb = tf.concat([predict_sims_emb, cur_sims_emb], axis=1)

        predict_sims_emb = predict_sims_emb[:, 1:]  # batch_size, sims_len * num_units
        predict_sims_emb = tf.reshape(predict_sims_emb,
                                      [-1, self.sims_len, self.num_units])  # batch_size, sims_len , num_units

        self.predict_target_emb = predict_target_lambda_emb  #
        self.predict_sims_emb = predict_sims_emb

        with tf.variable_scope('prepare_emb'):
            emb_for_type = self.predict_target_emb
            emb_for_time = h_i_minus

        with tf.variable_scope('intensity_calculation',reuse=tf.AUTO_REUSE):
            intensity_model = nhp_intensity_calculation( )

            self.target_lambda = intensity_model.cal_target_intensity(hidden_emb=self.predict_target_emb,
                                                                      type_num=self.type_num)
            self.sims_lambda = intensity_model.cal_sims_intensity(hidden_emb=self.predict_sims_emb,
                                                                  sims_len=self.sims_len,
                                                                  type_num=self.type_num)


        with tf.variable_scope('type_time_calculation',reuse=tf.AUTO_REUSE):
            time_predictor = thp_time_predictor()
            self.predict_time = time_predictor.predict_time(emb=emb_for_time,
                                                            num_units=self.num_units,)

            type_predictor = thp_type_predictor()
            self.predict_type_prob = type_predictor.predict_type(emb=emb_for_type,
                                                                 num_units=self.num_units,
                                                                 type_num=self.type_num)
            # self.predict_type_prob = tf.matmul(self.predict_type_prob, self.embedding.type_emb_lookup_table[:-3, :],
            #                                    transpose_b=True)
        self.output()



