# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 14:28
# @Author  : zxl
# @FileName: get_origin_data_brightkite.py


from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
from datetime import datetime
import time
np.random.seed(1234)


class Get_fs_data(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_fs_data, self).__init__(FLAGS = FLAGS)

        self.raw_data_path1 = "data/raw_data/fs/dataset_TSMC2014_NYC.csv"
        self.raw_data_path2 = "data/raw_data/fs/dataset_TSMC2014_TKY.csv"

        self.data_path = "data/orgin_data/fs.csv"

        if FLAGS.init_origin_data == True:

            self.get_fs_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def transform_time(self,x):
        dtime = datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y')
        timestamp = int(dtime.timestamp())
        return timestamp

    def normalize_data(self,df):
        df = df.rename(columns={'userId':'user_id','venueId':'item_id','venueCategory':'cat_id','utcTimestamp':'time_stamp'})

        df["time_stamp"] = df["time_stamp"].apply(lambda x: self.transform_time(x))
        return df

    def get_fs_data(self):


        df1 = pd.read_csv(self.raw_data_path1)
        df2 = pd.read_csv(self.raw_data_path2)

        df1 = self.normalize_data(df1)
        df2 = self.normalize_data(df2)
        df = pd.concat([df1,df2])

        location_df = self.filter(df)


        location_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data =  location_df
        print("well done!!")


