from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
from _datetime import datetime
import time
np.random.seed(1234)
from config.model_parameter import model_parameter


class Get_tmall_buy_data(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_tmall_buy_data, self).__init__(FLAGS = FLAGS)
        self.data_path = "data/orgin_data/tmall_buy.csv"

        if FLAGS.init_origin_data == True:
            self.raw_data_path = "data/raw_data/tmall_buy/tmall-buy.inter"
            self.get_tmall_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_tmall_data(self):

        with open(self.raw_data_path, 'r') as fin:
            # 确保字段相同
            resultList = []
            error_n = 0
            for line in fin:
                if 'user_id:token' in line:
                    continue
                try:
                    line = line.replace('\n', '')
                    arr = line.split('\t')
                    resultDic = {}
                    resultDic["user_id"] = arr[0]
                    resultDic["item_id"] = arr[2]
                    resultDic["time_stamp"] = int(arr[4])
                    resultDic["cat_id"] = 3
                    resultList.append(resultDic)
                except Exception as e:
                    error_n += 1
                    # self.logger.info("Error！！！！！！！！！！！！")
                    # self.logger.info(e)
            self.logger.info("total error entries" + str(error_n))

            location_df = pd.DataFrame(resultList)

        location_df = self.filter(location_df)

        location_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data = location_df
        print("well done!!")





