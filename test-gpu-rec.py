# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 11:04
# @Author  : zxl
# @FileName: train_process_check.py


import time
import random
import traceback
import numpy as np
import tensorflow.compat.v1 as tf


from config.model_parameter import model_parameter

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



    def train(self):


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
        print(self.FLAGS.cuda_visible_devices)

        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

if __name__ == '__main__':
    main_process = Train_main_process()
    main_process.train()
