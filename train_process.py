# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 9:26
# @Author  : zxl
# @FileName: train_process_no_time.py


import time
import random
import numpy as np
import traceback
import tensorflow.compat.v1 as tf

from DataHandle.get_origin_data_amazon_music import Get_amazon_data_music
from DataHandle.get_origin_data_taobao import Get_taobaoapp_data
from Embedding.Behavior_embedding_time_aware_attention import Behavior_embedding_time_aware_attention
from Model.BPRMF import BPRMF
from Model.hybird_baseline_models import NARM, NARM_time_att, NARM_time_att_time_rnn, LSTUR, LSTUR_time_rnn, STAMP,NARM2
from util.model_log import create_log
from DataHandle.get_input_data import DataInput
from Prepare.prepare_data_base import prepare_data_base
from DataHandle.get_origin_data_yoochoose import Get_yoochoose_data
from DataHandle.get_origin_data_movielen import Get_movie_data
from DataHandle.get_origin_data_tmall import Get_tmall_data
from DataHandle.get_origin_data_amazon_movie_tv import Get_amazon_data_movie_tv
from DataHandle.get_origin_data_amazon_elec import Get_amazon_data_elec
from DataHandle.get_origin_data_amazon_beauty import Get_amazon_data_beauty
from DataHandle.get_origin_data_brightkite import Get_BrightKite_data
from DataHandle.get_origin_data_order import Get_Order_data
from config.model_parameter import model_parameter
from Model.PISTRec_model import Time_Aware_self_Attention_model
from Model.attention_baseline_models import Self_Attention_Model, Time_Aware_Self_Attention_Model, \
    Ti_Self_Attention_Model
from Model.RNN_baesline_models import Gru4Rec, Vallina_Gru4Rec

from Model.RNN_baesline_models import  Gru4Rec,T_SeqRec
from Model.MTAMRec_model import MTAM, MTAM_via_T_GRU, MTAM_no_time_aware_rnn, \
    MTAM_no_time_aware_att, MTAM_hybird, MTAM_only_time_aware_RNN, MTAM_via_rnn, \
    MTAM_with_T_SeqRec,T_Gru_Extend
from Model.CLSTMRec import CLSTM,LSTM,MyLSTM,CGru
# from Model.GNNRec import GNN_T_Gru,GNN_T_Att,TimeAwareSR_GNN
from Model.gnn_baseline_model import SR_GNN
from Model.GRNNRec import GatedGrnnRec,AttentiveGrnnRec,GatRnnRec,OrderedGatRnnRec

import os
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)


class Train_main_process:

    def __init__(self):

        start_time = time.time()
        model_parameter_ins = model_parameter()
        experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
        self.FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS


        log_ins = create_log(type = self.FLAGS.type, experiment_type = self.FLAGS.experiment_type,
                             version=self.FLAGS.version)

        self.logger = log_ins.logger
        self.logger.info("hello world the experiment begin")

        # logger.info("The model parameter is :" + str(self.FLAGS._parse_flags()))

        if self.FLAGS.type == "yoochoose":
            get_origin_data_ins =  Get_yoochoose_data(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()


        elif self.FLAGS.type == "movielen":
            get_origin_data_ins =  Get_movie_data(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()


        if self.FLAGS.type == "tmall":
            get_origin_data_ins =  Get_tmall_data(FLAGS=self.FLAGS)


        elif self.FLAGS.type == "movie_tv":
            get_origin_data_ins =  Get_amazon_data_movie_tv(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()


        elif self.FLAGS.type == "elec":
            get_origin_data_ins =  Get_amazon_data_elec(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()

        elif self.FLAGS.type == "music":
            get_origin_data_ins =  Get_amazon_data_music(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()

        elif self.FLAGS.type == 'taobaoapp':
            get_origin_data_ins = Get_taobaoapp_data(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()

        elif self.FLAGS.type == "beauty":
            get_origin_data_ins =  Get_amazon_data_beauty(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()

        elif self.FLAGS.type == "brightkite":
            get_origin_data_ins =  Get_BrightKite_data(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()
        elif self.FLAGS.type == "order":
            get_origin_data_ins =  Get_Order_data(FLAGS=self.FLAGS)
            get_origin_data_ins.getDataStatistics()

        #get_train_test_ins = Get_train_test(FLAGS=self.FLAGS,origin_data=get_origin_data_ins.origin_data)
        prepare_data_behavior_ins = prepare_data_base(self.FLAGS,get_origin_data_ins.origin_data)
        self.train_set, self.test_set,self.dev_set = prepare_data_behavior_ins.get_train_test()

        #fetch part of test_data
        #if len(self.train_set) > 2000000:
            #self.test_set = random.sample(self.train_set,2000000)
            #self.test_set = self.test_set.sample(3500)

        self.logger.info('DataHandle Process.\tCost time: %.2fs' % (time.time() - start_time))
        start_time = time.time()



        self.emb = Behavior_embedding_time_aware_attention(is_training = self.FLAGS.is_training,
                                                           user_count  = prepare_data_behavior_ins.user_count,
                                                           item_count  = prepare_data_behavior_ins.item_count,
                                                           category_count= prepare_data_behavior_ins.category_count,
                                                           max_length_seq = self.FLAGS.length_of_user_history
                                                           )


        self.logger.info('Get Train Test Data Process.\tCost time: %.2fs' % (time.time() - start_time))

        self.item_category_dic = prepare_data_behavior_ins.item_category_dic
        self.global_step = 0
        self.one_epoch_step = 0
        self.now_epoch = 0


    def train(self):

        start_time = time.time()

        # Config GPU options
        if self.FLAGS.per_process_gpu_memory_fraction == 0.0:
            gpu_options = tf.GPUOptions(allow_growth=True)
        elif self.FLAGS.per_process_gpu_memory_fraction == 1.0:
            gpu_options = tf.GPUOptions()

        else:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.FLAGS.per_process_gpu_memory_fraction,allow_growth=True)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.FLAGS.cuda_visible_devices

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        if not tf.test.gpu_device_name():
            self.logger.warning("No GPU is found")
        else: self.logger.info(tf.test.gpu_device_name())

        global_step_lr = tf.Variable(0, trainable=False)
        lr1 = tf.train.exponential_decay(
            learning_rate=self.FLAGS.learning_rate, global_step=global_step_lr, decay_steps=1000, decay_rate=0.995, staircase=True)

        with self.sess.as_default():


            # Create a new model or reload existing checkpoint
            if self.FLAGS.experiment_type == "Vallina_Gru4Rec":
                self.model = Vallina_Gru4Rec(self.FLAGS, self.emb, self.sess)
            #1 baseline for all models
            if self.FLAGS.experiment_type == "Gru4Rec":
                self.model = Gru4Rec(self.FLAGS, self.emb, self.sess)
            #2
            elif self.FLAGS.experiment_type == "T_SeqRec":
                self.model = T_SeqRec(self.FLAGS, self.emb, self.sess)
            #3
            elif self.FLAGS.experiment_type == "NARM":
                self.model = NARM(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.experiment_type == "NARM2":
                self.model = NARM2(self.FLAGS, self.emb, self.sess)
            #4
            elif self.FLAGS.experiment_type == "NARM+":
                self.model = NARM_time_att(self.FLAGS, self.emb, self.sess)
            #5
            elif self.FLAGS.experiment_type == "NARM++":
                self.model = NARM_time_att_time_rnn(self.FLAGS, self.emb, self.sess)
            #6
            elif self.FLAGS.experiment_type == "LSTUR":
                self.model = LSTUR(self.FLAGS, self.emb, self.sess)
            #7
            elif self.FLAGS.experiment_type == "STAMP":
                self.model = STAMP(self.FLAGS, self.emb, self.sess)
            #8 the proposed model
            elif self.FLAGS.experiment_type == 'MTAM':
                self.model = MTAM(self.FLAGS, self.emb, self.sess)
            # 9
            elif self.FLAGS.experiment_type == "MTAM_no_time_aware_rnn":
                self.model = MTAM_no_time_aware_rnn(self.FLAGS, self.emb, self.sess)
            # 10
            elif self.FLAGS.experiment_type == "MTAM_no_time_aware_att":
                self.model = MTAM_no_time_aware_att(self.FLAGS, self.emb, self.sess)
            #11
            elif self.FLAGS.experiment_type == 'MTAM_via_T_GRU':
                self.model = MTAM_via_T_GRU(self.FLAGS, self.emb, self.sess)
            # 12
            elif self.FLAGS.experiment_type == 'MTAM_via_rnn':
                self.model = MTAM_via_rnn(self.FLAGS, self.emb, self.sess)
            #13 the proposed model
            elif self.FLAGS.experiment_type == 'T_GRU':
                self.model = MTAM_only_time_aware_RNN(self.FLAGS, self.emb, self.sess)
            # 14 the proposed model
            elif self.FLAGS.experiment_type == 'SASrec':
                self.model = Self_Attention_Model(self.FLAGS, self.emb, self.sess)
            # 15 the proposed model
            elif self.FLAGS.experiment_type == 'Time_Aware_Self_Attention_Model':
                self.model = Time_Aware_Self_Attention_Model(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.experiment_type == 'Ti_Self_Attention_Model':
                self.model = Ti_Self_Attention_Model(self.FLAGS, self.emb, self.sess)
            # 16 the proposed model
            elif self.FLAGS.experiment_type == 'bpr':
                self.model = BPRMF(self.FLAGS, self.emb, self.sess)
            # 17 the proposed model
            elif self.FLAGS.experiment_type == "MTAM_with_T_SeqRec":
                self.model = MTAM_with_T_SeqRec(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.experiment_type == "clstm":
                self.model = CLSTM(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.experiment_type == "lstm":
                self.model = LSTM(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.experiment_type == "mylstm":
                self.model = MyLSTM(self.FLAGS, self.emb, self.sess)

            elif self.FLAGS.experiment_type == "cgru":
                self.model = CGru(self.FLAGS, self.emb, self.sess)
            elif self.FLAGS.experiment_type == "T_Gru_Extend":
                self.model = T_Gru_Extend(self.FLAGS, self.emb, self.sess)


            self.logger.info('Init finish.\tCost time: %.2fs' % (time.time() - start_time))



            test_start = time.time()
            self.hr_1, self.ndcg_1, self.mrr_1,self.hr_5, self.ndcg_5, self.mrr_5, \
            self.hr_10, self.ndcg_10,self.mrr_10, self.hr_20, self.ndcg_20,self.mrr_20, \
            self.hr_50, self.ndcg_50, self.mrr_50 = \
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            self.best_dev_hr_10 = 0
            self.best_dev_ndcg_10 = 0

            self.best_result_hr = []
            self.best_result_ndcg = []

            self.max_stay_count = 0
            self.last_dev_hr_10 = 0
            self.last_dev_ndcg_10 = 0




            def eval_topk():



                dev_sum_hr_1, dev_sum_ndcg_1, dev_sum_mrr_1, dev_sum_hr_5, dev_sum_ndcg_5, dev_sum_mrr_5, \
                dev_sum_hr_10, dev_sum_ndcg_10,  dev_sum_mrr_10, dev_sum_hr_20, dev_sum_ndcg_20, dev_sum_mrr_20,\
                dev_sum_hr_50, dev_sum_ndcg_50, dev_sum_mrr_50 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                result_list_hr_all = []
                result_list_ndcg_all = []

                max_step = 0

                for step_i, batch_data in DataInput(self.dev_set, self.FLAGS.test_batch_size):
                    max_step = 1 + max_step
                    if self.FLAGS.experiment_type == "NARM" or \
                            self.FLAGS.experiment_type == "NARM+" or \
                            self.FLAGS.experiment_type == "NARM++" or \
                            self.FLAGS.experiment_type == "SR_GNN":
                        hr_1, ndcg_1, mrr_1, hr_5, ndcg_5, mrr_5,\
                        hr_10, ndcg_10, mrr_10, hr_20, ndcg_20, mrr_20, \
                        hr_50, ndcg_50, mrr_50,\
                        result_list_hr, result_list_ndcg = \
                            self.model.metrics_topK_concat(sess=self.sess,
                                                           batch_data=batch_data,
                                                           global_step=self.global_step,
                                                           topk=self.FLAGS.top_k)
                    else:
                        hr_1, ndcg_1, mrr_1, hr_5, ndcg_5, mrr_5,\
                        hr_10, ndcg_10, mrr_10, hr_20, ndcg_20, mrr_20,\
                        hr_50, ndcg_50, mrr_50,\
                        result_list_hr, result_list_ndcg = \
                            self.model.metrics_topK(sess=self.sess,
                                                    batch_data=batch_data,
                                                    global_step=self.global_step,
                                                    topk=self.FLAGS.top_k)
                    dev_sum_hr_1 = dev_sum_hr_1 + hr_1
                    dev_sum_ndcg_1 = dev_sum_ndcg_1 + ndcg_1
                    dev_sum_mrr_1 = dev_sum_mrr_1 + mrr_1
                    dev_sum_hr_5 = dev_sum_hr_5 + hr_5
                    dev_sum_ndcg_5 = dev_sum_ndcg_5 + ndcg_5
                    dev_sum_mrr_5 = dev_sum_mrr_5 + mrr_5
                    dev_sum_hr_10 = dev_sum_hr_10 + hr_10
                    dev_sum_ndcg_10 = dev_sum_ndcg_10 + ndcg_10
                    dev_sum_mrr_10 = dev_sum_mrr_10 + mrr_10
                    dev_sum_hr_20 = dev_sum_hr_20 + hr_20
                    dev_sum_ndcg_20 = dev_sum_ndcg_20 + ndcg_20
                    dev_sum_mrr_20 = dev_sum_mrr_20 + mrr_20
                    dev_sum_hr_50 = dev_sum_hr_50 + hr_50
                    dev_sum_ndcg_50 = dev_sum_ndcg_50 + ndcg_50
                    dev_sum_mrr_50 = dev_sum_mrr_50 + mrr_50


                dev_sum_hr_1 /= max_step
                dev_sum_ndcg_1 /= max_step
                dev_sum_mrr_1 /= max_step
                dev_sum_hr_5 /= max_step
                dev_sum_ndcg_5 /= max_step
                dev_sum_mrr_5 /= max_step
                dev_sum_hr_10 /= max_step
                dev_sum_ndcg_10 /= max_step
                dev_sum_mrr_10 /= max_step
                dev_sum_hr_20 /= max_step
                dev_sum_ndcg_20 /= max_step
                dev_sum_mrr_20 /= max_step
                dev_sum_hr_50 /= max_step
                dev_sum_ndcg_50 /= max_step
                dev_sum_mrr_50 /= max_step


                sum_hr_1, sum_ndcg_1, sum_mrr_1, sum_hr_5, sum_ndcg_5, sum_mrr_5,\
                sum_hr_10, sum_ndcg_10, sum_mrr_10, sum_hr_20, sum_ndcg_20, sum_mrr_20, \
                sum_hr_50, sum_ndcg_50, sum_mrr_50 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0,0,0,0
                result_list_hr_all = []
                result_list_ndcg_all = []

                max_step=0

                for step_i, batch_data in DataInput(self.test_set, self.FLAGS.test_batch_size):
                    max_step =1+max_step
                    if self.FLAGS.experiment_type == "NARM" or \
                            self.FLAGS.experiment_type == "NARM+" or \
                            self.FLAGS.experiment_type == "NARM++"   or \
                            self.FLAGS.experiment_type == "SR_GNN" :
                        hr_1, ndcg_1, mrr_1, hr_5, ndcg_5, mrr_5, \
                        hr_10, ndcg_10, mrr_10, hr_20, ndcg_20, mrr_20, \
                        hr_50, ndcg_50, mrr_50,\
                        result_list_hr, result_list_ndcg= \
                                                          self.model.metrics_topK_concat(sess=self.sess,
                                                                                         batch_data=batch_data,
                                                                                         global_step=self.global_step,
                                                                                         topk=self.FLAGS.top_k)
                    else:
                        hr_1, ndcg_1, mrr_1, hr_5, ndcg_5, mrr_5, \
                        hr_10, ndcg_10,  mrr_10, hr_20, ndcg_20, mrr_20,\
                        hr_50, ndcg_50, mrr_50,\
                        result_list_hr, result_list_ndcg = \
                                                          self.model.metrics_topK(sess=self.sess,
                                                                                  batch_data=batch_data,
                                                                                  global_step=self.global_step,
                                                                                  topk=self.FLAGS.top_k)
                    sum_hr_1 = sum_hr_1+hr_1
                    sum_ndcg_1 = sum_ndcg_1 + ndcg_1
                    sum_mrr_1 = sum_mrr_1 + mrr_1
                    sum_hr_5 = sum_hr_5 + hr_5
                    sum_ndcg_5 = sum_ndcg_5 + ndcg_5
                    sum_mrr_5 = sum_mrr_5 + mrr_5

                    sum_hr_10 = sum_hr_10 + hr_10
                    sum_ndcg_10 = sum_ndcg_10 + ndcg_10
                    sum_mrr_10 = sum_mrr_10 + mrr_10
                    sum_hr_20 = sum_hr_20 + hr_20
                    sum_ndcg_20 = sum_ndcg_20 + ndcg_20
                    sum_mrr_20 = sum_mrr_20 + mrr_20

                    sum_hr_50 = sum_hr_50 + hr_50
                    sum_ndcg_50  = sum_ndcg_50  + ndcg_50
                    sum_mrr_50 = sum_mrr_50 + mrr_50
                    result_list_hr_all= result_list_hr_all+result_list_hr
                    result_list_ndcg_all= result_list_ndcg_all+result_list_ndcg



                sum_hr_1 /= max_step
                sum_ndcg_1/= max_step
                sum_mrr_1 /= max_step
                sum_hr_5 /= max_step
                sum_ndcg_5 /= max_step
                sum_mrr_5 /= max_step
                sum_hr_10 /= max_step
                sum_ndcg_10 /= max_step
                sum_mrr_10 /= max_step
                sum_hr_20/= max_step
                sum_ndcg_20/= max_step
                sum_mrr_20 /= max_step
                sum_hr_50 /= max_step
                sum_ndcg_50 /= max_step
                sum_mrr_50 /= max_step

                if dev_sum_hr_10> self.best_dev_hr_10 and dev_sum_ndcg_10 > self.best_dev_ndcg_10:
                    self.best_dev_hr_10 = dev_sum_hr_10
                    self.best_dev_ndcg_10 = dev_sum_ndcg_10

                    self.hr_1, self.ndcg_1, self.mrr_1 = sum_hr_1, sum_ndcg_1, sum_mrr_1
                    self.hr_5, self.ndcg_5, self.mrr_5 = sum_hr_5, sum_ndcg_5, sum_mrr_5
                    self.hr_10, self.ndcg_10, self.mrr_10 = sum_hr_10 , sum_ndcg_10, sum_mrr_10
                    self.best_result_hr = result_list_hr_all
                    self.best_result_ndcg = result_list_ndcg_all
                    self.hr_20, self.ndcg_20,self.mrr_20 = sum_hr_20, sum_ndcg_20, sum_mrr_20
                    self.hr_50, self.ndcg_50, self.mrr_50 = sum_hr_50, sum_ndcg_50, sum_mrr_50


                def dev_log(k,hr,ndcg,mrr):

                    self.logger.info('Dev recall rate @ %d : %.4f   ndcg @ %d: %.4f mrr @ %d: %.4f' % (k , hr, k, ndcg,k,mrr))

                dev_log(1,dev_sum_hr_1,dev_sum_ndcg_1,dev_sum_mrr_1)
                dev_log(5, dev_sum_hr_5, dev_sum_ndcg_5,dev_sum_mrr_5)
                dev_log(10, dev_sum_hr_10, dev_sum_ndcg_10,dev_sum_mrr_10)
                dev_log(20, dev_sum_hr_20, dev_sum_ndcg_20,dev_sum_mrr_20)
                dev_log(50, dev_sum_hr_50, dev_sum_ndcg_50,dev_sum_mrr_50)

                def summery(k,hr,ndcg,mrr):
                    tag_recall = 'test recall@'+str(k)
                    tag_ndcg = 'test ndgc@'+str(k)
                    summary_recall_rate = tf.Summary(value=[tf.Summary.Value(tag=tag_recall, simple_value=hr)])
                    self.model.train_writer.add_summary(summary_recall_rate, global_step=self.global_step)
                    summary_avg_ndcg = tf.Summary(value=[tf.Summary.Value(tag=tag_ndcg, simple_value=ndcg)])
                    self.model.train_writer.add_summary(summary_avg_ndcg, global_step=self.global_step)
                    self.logger.info('Test recall rate @ %d : %.4f   ndcg @ %d: %.4f  mrr @ %d: %.4f' % (k , hr, k, ndcg,k,mrr))

                summery(1,sum_hr_1,sum_ndcg_1,sum_mrr_1)
                summery(5, sum_hr_5, sum_ndcg_5,sum_mrr_5)
                summery(10, sum_hr_10, sum_ndcg_10,sum_mrr_10)
                summery(20, sum_hr_20, sum_ndcg_20,sum_mrr_20)
                summery(50, sum_hr_50, sum_ndcg_50,sum_mrr_50)


            eval_topk()
            self.logger.info('End test. \tTest Cost time: %.2fs' % (time.time() - test_start))

            # Start training

            self.logger.info('Training....\tmax_epochs:%d\tepoch_size:%d' % (self.FLAGS.max_epochs,self.FLAGS.train_batch_size))
            start_time, avg_loss, self.best_auc,self.best_recall,self.best_ndcg = time.time(), 0.0,0.0,0.0,0.0
            for epoch in range(self.FLAGS.max_epochs):
                #if epoch > 2:
                    #lr = lr/1.5



                random.shuffle(self.train_set)
                self.logger.info('tain_set:%d'%len(self.train_set))
                epoch_start_time = time.time()
                learning_rate = self.FLAGS.learning_rate

                for step_i, train_batch_data in DataInput(self.train_set, self.FLAGS.train_batch_size):


                    # try:


                        #print(self.sess.run(global_step_lr))
                    # if learning_rate > 0.001:
                    learning_rate = self.sess.run(lr1,feed_dict={global_step_lr: self.global_step})
                    # else:
                    #     learning_rate = self.sess.run(lr2, feed_dict={global_step_lr: self.global_step})
                    #print(learning_rate)
                    add_summary = bool(self.global_step % self.FLAGS.display_freq == 0)
                    step_loss,merge = self.model.train(self.sess,train_batch_data,learning_rate,
                                                       add_summary,self.global_step,epoch)


                    self.sess.graph.finalize()
                    self.model.train_writer.add_summary(merge,self.global_step)
                    avg_loss = avg_loss + step_loss
                    self.global_step = self.global_step + 1
                    self.one_epoch_step = self.one_epoch_step + 1

                    #evaluate for eval steps
                    if self.global_step % self.FLAGS.eval_freq == 0:
                        print(learning_rate)
                        self.logger.info("Epoch step is " +  str(self.one_epoch_step))
                        self.logger.info("Global step is " +  str(self.global_step))
                        self.logger.info("Train_loss is " +  str(avg_loss / self.FLAGS.eval_freq))
                        # train_auc = self.model.metrics(sess=self.sess, batch_data=train_batch_data,
                        #                               global_step=self.global_step,name='train auc')
                        # self.logger.info('Batch Train AUC: %.4f' % train_auc)
                        # self.test_auc = self.model.metrics(sess=self.sess, batch_data=self.test_set,
                        #                               global_step=self.global_step,name='test auc')
                        # self.logger.info('Test AUC: %.4f' % self.test_auc)

                        eval_topk()
                        avg_loss = 0

                        self.save_model()
                        if self.FLAGS.draw_pic == True:
                            self.save_fig()

                    # except Exception as e:
                    #     self.logger.info("Error！！！！！！！！！！！！")
                    #     self.logger.info(e)
                    #     traceback.print_exc()


                self.logger.info('one epoch Cost time: %.2f' %(time.time() - epoch_start_time))
                self.logger.info("Epoch step is " + str(self.one_epoch_step))
                self.logger.info("Global step is " + str(self.global_step))
                self.logger.info("Train_loss is " + str(step_loss))

                eval_topk()
                with open('best_result_hr_'+self.FLAGS.version, 'w+') as f:
                    f.write(str(self.best_result_hr))
                with open('best_result_ndcg'+self.FLAGS.version, 'w+') as f:
                    f.write(str(self.best_result_ndcg))
                self.logger.info('Max recall rate @ 1: %.4f   ndcg @ 1: %.4f mrr @ 1: %.4f' % (self.hr_1, self.ndcg_1,self.mrr_1))
                self.logger.info('Max recall rate @ 5: %.4f   ndcg @ 5: %.4f mrr @ 1: %.4f' % (self.hr_5, self.ndcg_5, self.mrr_5))
                self.logger.info('Max recall rate @ 10: %.4f   ndcg @ 10: %.4f mrr @ 1: %.4f' % (self.hr_10, self.ndcg_10, self.mrr_10))
                self.logger.info('Max recall rate @ 20: %.4f   ndcg @ 20: %.4f mrr @ 1: %.4f' % (self.hr_20, self.ndcg_20,self.mrr_20 ))
                self.logger.info('Max recall rate @ 50: %.4f   ndcg @ 50: %.4f mrr @ 1: %.4f' % (self.hr_50, self.ndcg_50, self.mrr_50))


                if self.best_dev_hr_10 == self.last_dev_hr_10 and self.best_dev_ndcg_10 == self.last_dev_ndcg_10:
                    self.max_stay_count+=1
                else:
                    self.last_dev_hr_10 = self.best_dev_hr_10
                    self.last_dev_ndcg_10 = self.best_dev_ndcg_10
                    self.max_stay_count = 0
                if self.max_stay_count > 5:
                    break


                self.one_epoch_step = 0
                #if self.global_step > 1000:
                    #lr = lr / 2
                #if lr < 0.0005:
                    #lr = lr * 0.99
                #elif self.FLAGS.type == "tmall":
                    #lr = lr * 0.5
                #else:
                    #lr = lr * 0.98

                self.logger.info('Epoch %d DONE\tCost time: %.2f' %
                      (self.now_epoch, time.time() - start_time))

                self.now_epoch = self.now_epoch + 1
                self.one_epoch_step = 0


        # self.model.save(self.sess,self.global_step)
        self.logger.info('best test_auc: ' + str(self.best_auc))
        self.logger.info('best recall: ' + str(self.best_recall))

        self.logger.info('Finished')

    #judge to save model
    #three result for evaluating model: auc ndcg recall
    def save_model(self):

        #  avg_loss / self.FLAGS.eval_freq, test_auc,test_auc_new))
        # result.append((self.model.global_epoch_step.eval(), model.global_step.eval(), avg_loss / FLAGS.eval_freq, _eval(sess, test_set, model), _eval_auc(sess, test_set, model)))avg_loss = 0.0
        # only store good model

        is_save_model = False
        #for bsbe
        '''
        if self.FLAGS.experiment_type == "bsbe" or self.FLAGS.experiment_type == "bpr":
            if (self.test_auc > 0.85 and self.test_auc - self.best_auc > 0.01):
                self.best_auc = self.test_auc
                is_save_model = True
        #recall  for  istsbp
        elif self.FLAGS.experiment_type == "istsbp" or self.FLAGS.experiment_type == "pistrec":
            if self.recall_rate > 0.15 and self.recall_rate > self.best_recall:
                self.best_recall = self.recall_rate
                is_save_model =True
        '''

        # if self.global_step % 50000 == 0:
        #     is_save_model = True
        #
        # if is_save_model == True:
        #     self.model.save(self.sess, self.global_step)

    def save_fig(self):

        # save fig
        if self.global_step % (self.FLAGS.eval_freq * 1) == 0:
            input_dic = self.emb.make_feed_dic(batch_data=self.test_set[:100])
            if self.FLAGS.experiment_type == "istsbp" or \
                    self.FLAGS.experiment_type == "bsbe":
                input_dic[self.model.now_bacth_data_size] = len(self.test_set[:100])

            behavior = self.sess.run(self.model.user_history_embedding_result_dense, input_dic)
            usr_h_fig, short_term_intent_fig, attention_result_fig, item_emb_lookup_table_fig = \
                self.sess.run([self.model.user_h, self.model.short_term_intent,
                               self.model.attention_result,
                               self.emb.item_emb_lookup_table], input_dic)



            # draw picture
            generate_pic_class_ins = generate_pic_class(init=False, user_h=usr_h_fig,
                                                        short_term_intent=short_term_intent_fig,
                                                        attention_result=attention_result_fig,
                                                        item_table=item_emb_lookup_table_fig,
                                                        item_category_dic=self.item_category_dic)

            generate_pic_class_ins.draw_picure(type=self.FLAGS.type,
                                               experiment_type=self.FLAGS.experiment_type,
                                               version=self.FLAGS.version,
                                               global_step=self.global_step)

            self.logger.info("save fig finish!!!")


if __name__ == '__main__':
    main_process = Train_main_process()
    main_process.train()