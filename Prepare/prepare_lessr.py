# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 10:59
# @Author  : zxl
# @FileName: prepare_lessr.py

import os

"""
将数据集处理为LESSR模型所需要的的输入
"""

def deal_data(inpath, out_path, max_len):

    w = open(out_path,'w')

    item_set = set()

    with open(inpath,'r') as f:
        for l in f:
            example = eval(l)
            item_set=item_set.union(set(example[1]))

            s = ','.join([str(x) for x in example[1]])

            w.write(s+'\n')

    w.close()
    return item_set



if __name__ =="__main__":

    in_dir = "/home/zxl/project/MTAM-t2/data/training_testing_data_copy_11_10/"
    out_dir = "/home/zxl/project/MTAM-t2/data/training_testing_data_lessr/"


    for filename in os.listdir(in_dir):
    # for filename in ['taobaoapp_time_item_based_unidirection']:

        in_train_test_root = in_dir+filename+'/'

        data_name  = filename[:-29]

        out_train_test_root = out_dir+data_name+'/'

        if not os.path.exists(out_train_test_root):
            os.makedirs(out_train_test_root)


        all_item_set = set()

        for train_test_file in ['train_data.txt','test_data.txt','dev_data.txt']:
        # for train_test_file in [  'dev_data.txt']:
            in_file_path = in_train_test_root+train_test_file

            out_file_path = out_train_test_root + train_test_file.split('_')[0]+'.txt'



            if 'train' in train_test_file:
                max_len = 10000000
            else:
                max_len = 20000

            single_item_set = deal_data(in_file_path,out_file_path,max_len)
            all_item_set=all_item_set.union(single_item_set)


        out_num_file = out_train_test_root+'num_items.txt'
        with open(out_num_file,'w') as w:
            w.write(str(len(all_item_set))+'\n')








