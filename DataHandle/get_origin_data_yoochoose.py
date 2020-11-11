from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import datetime
import time
np.random.seed(1234)
from config.model_parameter import model_parameter

class Get_yoochoose_data(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_yoochoose_data, self).__init__(FLAGS = FLAGS)
        self.data_path = "data/orgin_data/yoochoose.csv"
        if FLAGS.init_origin_data == True:
            self.yoochose_data = pd.read_csv("data/raw_data/yoochoose/yoochoose-clicks.dat",
                                             header=None,
                                             names=["user_id", "time_stamp", "item_id", "cat_id"])

            # 选出相应的字段
            self.yoochose_data = self.yoochose_data[["user_id", "time_stamp", "item_id", "cat_id"]]
            self.get_yoochoose_data()

        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_yoochoose_data(self):

        def transform_time(x):
            #去掉特殊符号
            x = x.split(".")[0]
            x = x.split("T")
            x = x[0]+" "+x[1]
            #只用小时
            #x = x.split(":")[0]
            x = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            x = time.mktime(x.timetuple())
            return x

        #进行拼接，进行格式的规范化
        self.yoochose_data["time_stamp"] = self.yoochose_data["time_stamp"].apply(lambda x:transform_time(x))

        #根据用户随机筛选出1/4
        user_filter = self.yoochose_data.groupby("user_id").count()
        userfiltered = user_filter.sample(frac = 0.25)
        self.yoochose_data_filter = self.yoochose_data[self.yoochose_data['user_id'].isin(userfiltered.index)]
        print(self.yoochose_data_filter.shape)

        self.yoochose_data_filter = self.filter(self.yoochose_data_filter)

        # # user sequence<3过滤

        self.yoochose_data_filter.to_csv(self.data_path,encoding="UTF8",index=False)
        self.origin_data = self.yoochose_data_filter


if __name__ == "__main__":

    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS

    ins = Get_yoochoose_data(FLAGS=FLAGS)
    ins.getDataStatistics()

    # print(origin_data)





