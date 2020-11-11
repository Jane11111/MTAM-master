from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
from _datetime import datetime
import time
np.random.seed(1234)
from config.model_parameter import model_parameter


class Get_tmall_data(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_tmall_data, self).__init__(FLAGS = FLAGS)
        self.data_path = "data/orgin_data/tmall.csv"

        if FLAGS.init_origin_data == True:
            self.raw_data_path = "data/raw_data/tianchi/user_log_format1.csv"
            self.get_tmall_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_tmall_data(self):

        reviews_df = pd.read_csv(self.raw_data_path)
        reviews_df = reviews_df[["user_id","item_id","cat_id","time_stamp"]]
        print(reviews_df.shape)

        #时间处理
        def time2stamp(time_s):
            try:
                time_s = '2015' + str(time_s)
                time_s = time.strptime(time_s, '%Y%m%d')
                stamp = int(time.mktime(time_s))
                return stamp
            except Exception as e:
                print(time_s)
                print(e)
        reviews_df["time_stamp"] = reviews_df["time_stamp"].apply(lambda x:time2stamp(x))
        # user sequence<3进行过滤

        user_filter = reviews_df.groupby("user_id").count()
        user_filter = user_filter[user_filter["item_id"]>=3]
        reviews_df_filter = reviews_df[reviews_df["user_id"].isin(user_filter.index)]
        print(reviews_df_filter.shape)
        reviews_df_filter.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data = reviews_df_filter


if __name__ == "__main__":

    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS

    ins = Get_tmall_data(FLAGS=FLAGS)
    ins.getDataStatistics()





    # print(origin_data)





