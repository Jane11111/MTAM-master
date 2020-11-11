# -*- coding: utf-8 -*-
# @Time    : 2020/10/5 19:33
# @Author  : zxl
# @FileName: test-reconsume.py

import numpy as np

if __name__ == "__main__":

    root = 'D:\Project\MTAM-master\data\\training_testing_data\yoochoose_time_item_based_unidirection\\'
    data_path = root+"train_data.txt"

    total_count = 0
    re_count = 0
    dic = {}

    with open(data_path,'r') as f:
        l = f.readline()
        while l:
            total_count+=1
            arr = eval(l)
            item_lst = arr[1]
            time_lst = arr[3]
            target_item = arr[7][0]
            target_time = arr[7][2]
            if target_item in item_lst:

                re_count +=1

                idx = np.argwhere(np.array(item_lst) == target_item)[-1][0]
                last_time = time_lst[idx]
                interval = target_time - last_time
                print("item: %d, interval: %d, positino interval: %d"%(target_item,interval,len(item_lst)-1-idx))

                if target_item not in dic:
                    dic[target_item]= []
                dic[target_item].append(len(item_lst)-1-idx)

            l = f.readline()
    for k in dic:
        print(str(k) + ":" + str(dic[k]))
    print("total count: %d, reconsume count: %d, re rate: %.5f"%(total_count,re_count,re_count/total_count))