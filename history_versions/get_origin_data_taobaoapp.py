from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import datetime
import time
np.random.seed(1234)


class Get_yoochoose_data(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_yoochoose_data, self).__init__(FLAGS = FLAGS)
        self.yoochose_data = pd.read_csv("data/raw_data/yoochoose/yoochoose-clicks.dat",
                                         header = None,
                                         names = ["user_id","timestamp","item_id","pricing"])

        self.data_path = "data/orgin_data/yoochoose.csv"

        if FLAGS.init_origin_data == True:
            self.get_movie_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_movie_data(self):

        def transform_time(x):
            #去掉特殊符号
            x = x.replace("T"," ").split(".")[0]
            x = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            x = time.mktime(x.timetuple())
            return x


        #进行拼接，进行格式的规范化
        self.yoochose_data["timestamp"] = self.yoochose_data["timestamp"].apply(lambda x:transform_time(x))
        self.yoochose_data.to_csv(self.data_path,encoding="UTF8",index=False)
        self.origin_data = self.yoochose_data



if __name__ == "__main__":

    get_data_ins = Get_yoochoose_data(type = "Tianchi")
    print("hello")


    # print(origin_data)





