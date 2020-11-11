# -*- coding: utf-8 -*-
# @Time    : 2020/10/7 19:46
# @Author  : zxl
# @FileName: interval_distr.py


import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root = 'D:\Project\MTAM-master\data\\training_testing_data\elec_time_item_based_unidirection\\'
    data_path = root + "train_data.txt"

    total_count = 0
    re_count = 0
    dic = {}
    interval_lst = []
    with open(data_path, 'r') as f:
        l = f.readline()
        while l:
            arr = eval(l)

            item_lst = arr[1]
            leng = arr[8]
            time_lst = arr[3]
            target_time = arr[7][2]
            target_item = arr[7][0]
            last_time = time_lst[leng-2]
            last_item = item_lst[leng-2]
            interval = target_time - last_time
            if last_item not in dic:
                dic[last_item] = []
            if interval < 20000:
                interval_lst.append(interval)
                dic[last_item].append(interval)


            l = f.readline()
    plt.hist(interval_lst)
    plt.show()

    count = 10
    for item in dic:
        # if count <0:
        #     break
        if len(dic[item])< 100:
            continue
        count-=1
        plt.hist(dic[item])
        plt.show()