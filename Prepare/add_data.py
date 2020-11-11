# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 16:49
# @Author  : zxl
# @FileName: add_data.py

import os

"""
给原来的书架上reconsume_lst和is_reconsume，在填充数据时候不用再处理
"""

if __name__ == "__main__":

    root = '/home/zxl/project/MTAM-t2/data/training_testing_data/'

    for dir in ['brightkite_time_item_based_unidirection']:
        dir_path = root+dir+'/'
        print(dir_path)

        for filename in ['train_data.txt','test_data.txt']:
            new_filename = filename[:-4]+'_new'+'.txt'
            old_filename = filename[:-4] + '_old' + '.txt'

            w_new = open(dir_path+new_filename,'w')
            w_old = open(dir_path + old_filename, 'w')
            with open(dir_path+filename,'r') as f:
                for l in f:
                    w_old.write(l)
                    l = l.replace('\n','')
                    example = list(eval(l))
                    item_list = example[1]
                    is_reconsume=float(example[7][0] in example[1])
                    reconsumes = []
                    for i in range(len(item_list )):
                        reconsumes.append(float(item_list [i] in item_list [:i]))
                    example.append(is_reconsume)
                    example.append(reconsumes)
                    w_new.write(str(example)+'\n')

            w_new.close()
            w_old.close()

            w=open(dir_path+filename,'w')
            with open(dir_path+new_filename,'r') as f:
                for l in f:
                    w.write(l)
            w.close()