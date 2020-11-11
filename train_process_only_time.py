# -*- coding: utf-8 -*-
# @Time    : 2020/10/19 10:36
# @Author  : zxl
# @FileName: train_process_only_time.py



import time
import random
import traceback
import numpy as np
import tensorflow.compat.v1 as tf


from DataHandle.get_origin_data_amazon_music import Get_amazon_data_music
from DataHandle.get_origin_data_taobao import Get_taobaoapp_data
from Embedding.Behavior_embedding_time_aware_attention import Behavior_embedding_time_aware_attention
from Model.BPRMF import BPRMF
from Model.hybird_baseline_models import NARM, NARM_time_att, NARM_time_att_time_rnn, LSTUR, LSTUR_time_rnn, STAMP
from util.model_log import create_log
from DataHandle.get_input_data import DataInput
from Prepare.prepare_data_base import prepare_data_base
from DataHandle.get_origin_data_yoochoose import Get_yoochoose_data
from DataHandle.get_origin_data_movielen import Get_movie_data
from DataHandle.get_origin_data_tmall import Get_tmall_data
from DataHandle.get_origin_data_amazon_movie_tv import Get_amazon_data_movie_tv
from DataHandle.get_origin_data_amazon_elec import Get_amazon_data_elec
from config.model_parameter import model_parameter
from Model.PISTRec_model import Time_Aware_self_Attention_model
from Model.attention_baseline_models import Self_Attention_Model, Time_Aware_Self_Attention_Model, \
    Ti_Self_Attention_Model
from Model.RNN_baesline_models import Gru4Rec, Vallina_Gru4Rec

from Model.RNN_baesline_models import  Gru4Rec,T_SeqRec
from Model.MTAMRec_model import MTAM, MTAM_via_T_GRU, MTAM_no_time_aware_rnn, \
    MTAM_no_time_aware_att, MTAM_hybird, MTAM_only_time_aware_RNN, MTAM_via_rnn, MTAM_with_T_SeqRec
from Model.TimePred_model import TimePred
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


        #get_train_test_ins = Get_train_test(FLAGS=self.FLAGS,origin_data=get_origin_data_ins.origin_data)
        prepare_data_behavior_ins = prepare_data_base(self.FLAGS,get_origin_data_ins.origin_data)
        self.train_set, self.test_set = prepare_data_behavior_ins.get_train_test()

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
        lr2 = tf.train.exponential_decay(
            learning_rate=0.001, global_step=global_step_lr, decay_steps=1000, decay_rate=0.995,
            staircase=True)

        def eval_mse():


            max_step = 0.
            mse_lst = []

            for step_i, batch_data in DataInput(self.test_set, self.FLAGS.test_batch_size):
                max_step = 1 + max_step

                step_mse =  self.model.metrics_mse(sess=self.sess,
                                            batch_data=batch_data,
                                            global_step=self.global_step,
                                            topk=self.FLAGS.top_k)
                mse_lst.extend(list(step_mse[0]))

            mse_val = np.mean(mse_lst)

            if mse_val < self.mse:
                self.mse = mse_val

            print('----test mse: %.5f-----'%mse_val)
            print('----MIN mse: %.5f-----'%(self.mse))
        with self.sess.as_default():



            self.model = TimePred(self.FLAGS,self.emb, self.sess)


            self.logger.info('Init finish.\tCost time: %.2fs' % (time.time() - start_time))


            test_start = time.time()
            self.mse = 100



            self.logger.info('End test. \tTest Cost time: %.2fs' % (time.time() - test_start))

            # Start training

            self.logger.info('Training....\tmax_epochs:%d\tepoch_size:%d' % (self.FLAGS.max_epochs,self.FLAGS.train_batch_size))
            start_time, avg_loss, self.best_auc,self.best_recall,self.best_ndcg = time.time(), 0.0,0.0,0.0,0.0
            loss_origin = []
            loss_time = []
            eval_mse()
            for epoch in range(self.FLAGS.max_epochs):
                #if epoch > 2:
                    #lr = lr/1.5



                random.shuffle(self.train_set)
                self.logger.info('tain_set:%d'%len(self.train_set))
                epoch_start_time = time.time()
                learning_rate = self.FLAGS.learning_rate

                for step_i, train_batch_data in DataInput(self.train_set, self.FLAGS.train_batch_size):
                    try:
                        #print(self.sess.run(global_step_lr))
                        if learning_rate > 0.001:
                            learning_rate = self.sess.run(lr1,feed_dict={global_step_lr: self.global_step})
                        else:
                            learning_rate = self.sess.run(lr2, feed_dict={global_step_lr: self.global_step})
                        #print(learning_rate)
                        add_summary = bool(self.global_step % self.FLAGS.display_freq == 0)
                        step_loss, step_loss_time, merge = self.model.train(self.sess,train_batch_data,learning_rate,
                                                           add_summary,self.global_step,epoch)

                        self.sess.graph.finalize()

                        self.model.train_writer.add_summary(merge,self.global_step)
                        avg_loss = avg_loss + step_loss
                        loss_time.extend(step_loss_time)
                        self.global_step = self.global_step + 1
                        self.one_epoch_step = self.one_epoch_step + 1

                        #evaluate for eval steps
                        if self.global_step % self.FLAGS.eval_freq == 0:
                            print(learning_rate)
                            self.logger.info("Epoch step is " +  str(self.one_epoch_step))
                            self.logger.info("Global step is " +  str(self.global_step))
                            self.logger.info("Train_loss is " +  str(avg_loss / self.FLAGS.eval_freq))
                            self.logger.info("Time Loss is "+ str(np.mean(np.array(loss_time))))
                            eval_mse()
                            avg_loss = 0
                            loss_origin = []
                            loss_time = []
                            loss_reconsume = []



                    except Exception as e:
                        self.logger.info("Error！！！！！！！！！！！！")
                        self.logger.info(e)
                        traceback.print_exc()


                self.logger.info('one epoch Cost time: %.2f' %(time.time() - epoch_start_time))
                self.logger.info("Epoch step is " + str(self.one_epoch_step))
                self.logger.info("Global step is " + str(self.global_step))
                self.logger.info("Train_loss is " + str(step_loss))
                self.logger.info("Time Loss is " + str(np.mean(np.array(loss_time))))
                eval_mse()
                self.one_epoch_step = 0


                self.logger.info('Epoch %d DONE\tCost time: %.2f' %
                      (self.now_epoch, time.time() - start_time))

                self.now_epoch = self.now_epoch + 1
                self.one_epoch_step = 0


        self.model.save(self.sess,self.global_step)
        self.logger.info('best test_auc: ' + str(self.best_auc))
        self.logger.info('best recall: ' + str(self.best_recall))

        self.logger.info('Finished')




if __name__ == '__main__':
    main_process = Train_main_process()
    main_process.train()
