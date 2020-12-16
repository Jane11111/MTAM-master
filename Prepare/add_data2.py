# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 16:49
# @Author  : zxl
# @FileName: add_data.py

import os

"""
增加邻接矩阵
"""
import numpy as np


def construct_graph( u_input):
    # 自环只加一次

    position_count = len(u_input)
    u_input = np.array(u_input)
    u_A_out = np.zeros(shape=(position_count, position_count) ,dtype=np.int)
    u_A_in = np.zeros(shape=(position_count, position_count), dtype=np.int)

    if len(u_input) == len(set(u_input)):
        for i in np.arange(len(u_input) - 1):
            u_A_out[i, i:] = 1
            u_A_in[i,:i] = 1
    else:
        # print(u_input)
        item2idx = {}
        for i in range(len(u_input)):
            item = u_input[i]
            lst = np.where(u_input == item)[0]
            item2idx[item] = lst


        processed = {}
        for i in np.arange(len(u_input) - 1):
            # u_lst = np.where(u_input == u_input[i])[0]
            u_lst = item2idx[u_input[i]]
            u_A_out[i, i:] = 1
            for j in np.arange(i, len(u_input), 1):
                tuple = (u_input[i], u_input[j])
                if tuple in processed:
                    continue
                processed[tuple] = True
                # v_lst = np.where(u_input == u_input[j])[0]
                v_lst = item2idx[u_input[j]]
                for u in u_lst:
                    for v in v_lst:
                        u_A_out[u,v] = 1  # 每个结点只计算一次
        processed = {}
        for i in np.arange(len(u_input)):
            # u_lst = np.where(u_input == u_input[i])[0]
            u_lst = item2idx[u_input[i]]
            for j in np.arange(0,i+1, 1):
                tuple = (u_input[i],u_input[j])
                if tuple in processed:
                    continue
                processed[tuple] = True
                # v_lst = np.where(u_input == u_input[j])[0]
                v_lst = item2idx[u_input[j]]
                for u in u_lst:
                    for v in v_lst:
                        u_A_in[u, v] = 1  # 每个结点只计算一次
    u_A_in = u_A_in.tolist()
    u_A_out=u_A_out.tolist()
    return u_A_in,u_A_out





def process_graph_matrix(  example):
    item_list = example[1][:-1]

    u_A_in,u_A_out  =  construct_graph(item_list )

    example.append(u_A_in)
    example.append( u_A_out)
    return example

if __name__ == "__main__":

    root = '/home/zxl/project/MTAM-t2/data/training_testing_data/'

    for dir in ['toys_time_item_based_unidirection']:
        dir_path = root+dir+'/'
        print(dir_path)


        for filename in ['train_data.txt','test_data.txt','dev_data.txt']:
            new_filename = filename[:-4]+'_new'+'.txt'
            i = 0
            w_new = open(dir_path+new_filename,'w')

            with open(dir_path+filename,'r') as f:
                for l in f:
                    i+=1
                    # if i > 5000:
                    #     break

                    line = list(eval(l))
                    line =  process_graph_matrix(line )
                    w_new.write(str(line)+'\n')

            w_new.close()
    # a = [1,2,3,1,5]
    # u_A_in, u_A_out = construct_graph(a)
    # print(u_A_in)
    # print(u_A_out)

