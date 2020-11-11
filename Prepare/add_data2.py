# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 16:49
# @Author  : zxl
# @FileName: add_data.py

import os

"""
增加邻接矩阵
"""
import numpy as np


def construct_graph ( u_input, t_input):
    position_count = len(u_input)
    u_input = np.array(u_input)
    u_A_out = np.zeros((position_count, position_count))
    u_A_in = np.zeros((position_count, position_count))

    t_u_A_out = np.zeros((position_count, position_count))
    t_u_A_in = np.zeros((position_count, position_count))
    #
    # in_masks = np.zeros((position_count, position_count))
    # out_masks = np.zeros((position_count,position_count))

    in_edge_ids = np.zeros((position_count, position_count))
    out_edge_ids = np.zeros((position_count, position_count))

    sum_time = np.zeros((position_count,))
    count_neighbor = np.zeros((position_count,))
    for i in np.arange(len(u_input) - 2):
        u_lst = np.where(u_input == u_input[i])[0]
        v_lst = np.where(u_input == u_input[i + 1])[0]
        origin_t = t_input[i + 1]
        t = np.exp(-t_input[i + 1])  # TODO 这个时间怎么处理

        for u in u_lst:
            sum_time[u] += origin_t
            count_neighbor[u] += 1
            for v in v_lst:
                u_A_out[u][v] += 1 / len(v_lst)
                out_edge_ids[u][v] = i
                t_u_A_out[u][v] = t


        for v in v_lst:
            sum_time[v] += origin_t
            count_neighbor[v] += 1
            for u in u_lst:
                u_A_in[v][u] += 1 / len(u_lst)
                in_edge_ids[v][u] = i
                t_u_A_in[v][u] = t

    u_sum_in = np.reshape(np.sum(u_A_in, 1), (-1, 1))
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(u_A_in, u_sum_in)

    u_sum_out = np.reshape(np.sum(u_A_out, 1), (-1, 1))
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(u_A_out, u_sum_out)

    count_neighbor[np.where(count_neighbor == 0)] = 1
    avg_time = sum_time / count_neighbor

    sparse_adj_in = []
    sparse_adj_out = []
    t_sparse_adj_in = []
    t_sparse_adj_out = []
    sparse_eid_adj_in = []
    sparse_eid_adj_out = []
    for i in range(len(u_input)):
        for j in range(len(u_input)):
            if u_A_in[i][j] !=0:
                sparse_adj_in.append([i,j,u_A_in[i][j]])
            if u_A_out[i][j] !=0:
                sparse_adj_out.append([i,j,u_A_out[i][j]])
            if t_u_A_in[i][j] != 0:
                sparse_adj_in.append([i, j, t_u_A_in[i][j]])
            if t_u_A_out[i][j] != 0:
                sparse_adj_out.append([i, j, t_u_A_out[i][j]])

            if in_edge_ids[i][j] !=0:
                sparse_eid_adj_in.append([i,j,in_edge_ids[i][j]])
            if out_edge_ids[i][j] !=0:
                sparse_eid_adj_out.append([i,j,out_edge_ids[i][j]])

    return sparse_adj_in, sparse_adj_out, t_sparse_adj_in, t_sparse_adj_out,sparse_eid_adj_in, sparse_eid_adj_out, list(avg_time)


def process_graph_matrix(  example):
    item_list = example[1]
    time_list = example[3]

    u_A_in, u_A_out, \
    t_sparse_adj_in, t_sparse_adj_out,\
    in_edge_ids, out_edge_ids, \
    avg_time =  construct_graph(item_list, time_list)

    example.extend([u_A_in, u_A_out, t_sparse_adj_in, t_sparse_adj_out,in_edge_ids,out_edge_ids,avg_time])
    return example

if __name__ == "__main__":

    root = '/home/zxl/project/MTAM-t2/data/training_testing_data/'

    for dir in ['music_time_item_based_unidirection']:
        dir_path = root+dir+'/'
        print(dir_path)


        for filename in ['train_data.txt','test_data.txt']:
            new_filename = filename[:-4]+'_new'+'.txt'
            i = 0
            w_new = open(dir_path+new_filename,'w')

            with open(dir_path+filename,'r') as f:
                for l in f:
                    i+=1
                    if i > 5000:
                        break

                    line = list(eval(l))
                    line =  process_graph_matrix(line )
                    w_new.write(str(line)+'\n')

            w_new.close()


            w=open(dir_path+filename,'w')
            with open(dir_path+new_filename,'r') as f:
                for l in f:
                    w.write(l)
            w.close()