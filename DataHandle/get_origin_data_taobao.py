from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import time
from config.model_parameter import model_parameter
np.random.seed(1234)

class Get_taobaoapp_data(Get_origin_data_base):
    def __init__(self,FLAGS):
        super(Get_taobaoapp_data,self).__init__(FLAGS = FLAGS)
        self.data_path = "data/orgin_data/taobaoapp.csv"

        if FLAGS.init_origin_data == True:
            self.taobaoapp_data = pd.read_csv("data/raw_data/taobaoapp/tianchi_mobile_recommend_train_user.csv")
            self.get_taobaoapp_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_taobaoapp_data(self):
        def transform_time(x):
            timeArray = time.strptime(x,"%Y-%m-%d %H")
            timeStamp = int(time.mktime(timeArray))
            return timeStamp

        self.taobaoapp_data = self.taobaoapp_data[["user_id", "item_id", "item_category", "time"]]
        self.taobaoapp_data = self.taobaoapp_data.rename(columns={"item_category": "cat_id",
                                                                "time": "time_stamp",
                                                                })
        self.taobaoapp_data["time_stamp"] = self.taobaoapp_data["time_stamp"].apply(lambda x:transform_time(x))
        #         对user sequence小于3的进行过滤
        self.filtered_data = self.filter(self.taobaoapp_data)

        # TODO for testing
        # self.filtered_data = self.taobaoapp_data

        self.filtered_data.to_csv(self.data_path, encoding="UTF8", index=False)
        self.origin_data = self.filtered_data


if __name__ == "__main__":
    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS

    ins = Get_taobaoapp_data(FLAGS=FLAGS)
    ins.getDataStatistics()

    # print(origin_data)

