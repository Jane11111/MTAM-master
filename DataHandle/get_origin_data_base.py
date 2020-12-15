"""
Coded by Wenxian and Yinglong
Reviewed by Wendy
"""
import pandas as pd
import os
from datetime import datetime
from sklearn import preprocessing
from util.model_log import create_log
from config.model_parameter import model_parameter


class Get_origin_data_base():

    '''
    Get_origin_data(type)
    process raw data and get original data. Input dataset name.
    Do statisitcs.
    '''


    def __init__(self, FLAGS):

        """ Choose Dataset and get the processed origin data file
            If the processed origin data does not exist, do get_type_data() and do statistics from
            raw data.
        :param type:
            type: "Tmall", "Amazon"
                Optional type of Datasets.
            origin_data_path : A String
            raw_data_path : A String
            raw_data_path_meta : A String
                for "Amazon"
        """


        log_ins = create_log()
        self.logger = log_ins.logger
        self.type = FLAGS.type






    #进行两者统计
    def getDataStatistics(self):

        reviews_df = self.origin_data
        user = set(reviews_df["user_id"].tolist())
        self.logger.info("The user count is " + str(len(user)))
        item = set(reviews_df["item_id"].tolist())
        self.logger.info("The item count is " + str(len(item)))
        category = set(reviews_df["cat_id"].tolist())
        self.logger.info("The category  count is " + str(len(category)))

        behavior = reviews_df.shape[0]
        self.logger.info("The behavior count is " + str(behavior))


        behavior_per_user = reviews_df.groupby(by=["user_id"], as_index=False)["item_id"].count()
        behavior_per_user = behavior_per_user["item_id"].mean()
        self.logger.info("The avg behavior of each user count is " + str(behavior_per_user))

        behavior_per_item = reviews_df.groupby(by=["item_id"], as_index=False)["user_id"].count()
        behavior_per_item = behavior_per_item["user_id"].mean()
        self.logger.info("The avg behavior of each item count is " + str(behavior_per_item))


    def top_user_purchase(self):
        result_list = []

        def get_top_item(x):
            print(x)

        self.origin_data.groupby(by=["user_id","item_id"], as_index=False).apply(lambda x:get_top_item(x))

    def filter(self, data):
        # filtering item < 5
        if self.type == 'wendy':
            user_filter = data.groupby("user_id").count()
            user_filter = user_filter[user_filter['item_id'] >= 2]
            data = data[data['user_id'].isin(user_filter.index)]

            item_filter = data.groupby("item_id").count()
            item_filter = item_filter[item_filter['user_id'] >= 2]
            data = data[
                data['item_id'].isin(item_filter.index)]
            # filtering user < 2

        else:
            item_filter = data.groupby("item_id").count()

            if self.type != 'taobaoapp':
                item_filter = item_filter[item_filter['user_id'] >= 30]
                #item_filter = item_filter[item_filter['user_id'] >= 10]
            else:
                item_filter = item_filter[item_filter['user_id'] >=50]


            data = data[
                data['item_id'].isin(item_filter.index)]
            # filtering user < 2
            user_filter = data.groupby("user_id").count()
            if self.type == 'elec' or self.type == 'order' or self.type == 'movie_tv':
                #user_filter = user_filter[user_filtermovielen['item_id'] >= 20]
                user_filter = user_filter[user_filter['item_id'] >= 20]
            else :
                user_filter = user_filter[user_filter['item_id'] >= 10]
            data = data[data['user_id'].isin(user_filter.index)]
        return data



if __name__ == "__main__":


    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS








