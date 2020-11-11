"""
Coded by Wenxian and Yinglong
Reviewed by Wendy
"""
import pandas as pd
import os
from datetime import datetime
from sklearn import preprocessing

class Get_origin_data():
    '''
    Get_origin_data(type)
    process raw data and get original data. Input dataset name.
    Do statisitcs.
    '''


    def __init__(self, type, raw_data_path = None, raw_data_path_meta = None,logger = None):
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
        self.logger = logger
        self.type = type
        if self.type == "Tianchi":
            self.filePath = 'data/orgin_data/tianchi_origin_data.csv'
            self.filePath = self.filePath
            if os.path.exists(self.filePath):
                self.origin_data = pd.read_csv(self.filePath)
            else:
                self.raw_data_path = raw_data_path
                self.origin_data = self.get_tianchi_data()
        if self.type == "Amazon":
            self.filePath = 'data/orgin_data/amazon_origin_data.csv'
            self.filePath = self.filePath
            if os.path.exists(self.filePath):
                self.origin_data = pd.read_csv(self.filePath)
            else:
                self.raw_data_path = raw_data_path
                self.raw_data_path_meta = raw_data_path_meta
                self.origin_data = self.get_amazon_data()


    def get_tianchi_data(self):
        #reviews_df = pd.read_csv("/Users/wendy/Downloads/data_format1/tianchi_data.csv")
        reviews_df = pd.read_csv(self.raw_data_path)
        #reviews_df = reviews_df.sample(100000)
        print(reviews_df.shape)

        resultItemList = []

        def GetMostItemFrency(x):
            if x.shape[0] >= 10:
                resultDic = {}
                resultDic["item_id"] = x["item_id"].tolist()[0]
                resultItemList.append(resultDic)

        reviews_df.groupby(["item_id"]).apply(lambda x: GetMostItemFrency(x))
        resultItemList = pd.DataFrame(resultItemList)
        #print(resultItemList)

        resultUserList = []

        def GetUserItemFrency(x):
            buyCount = x.loc[x["action_type"] == 2].shape[0]
            if x.shape[0] >= 20 and buyCount >= 5:
                resultDic = {}
                resultDic["user_id"] = x["user_id"].tolist()[0]
                resultUserList.append(resultDic)

        reviews_df.groupby(["user_id"]).apply(lambda x: GetUserItemFrency(x))
        resultUserList = pd.DataFrame(resultUserList)
        #print(resultUserList)

        reviews_df = pd.merge(reviews_df, resultItemList, on="item_id")
        reviews_df = pd.merge(reviews_df, resultUserList, on="user_id")

        def time2stamp(time_s):
            try:
                time_s = '2015' + str(time_s)
                time_s = datetime.strptime(time_s, '%Y%m%d')
                stamp = int(datetime.timestamp(time_s))
                return stamp
            except Exception as e:
                print(time_s)
                print(e)

        reviews_df['time_stamp'] = reviews_df['time_stamp'].apply(time2stamp)

        print(reviews_df.shape)
        reviews_df.to_csv(self.filePath, index=False, encoding="UTF8")
        return reviews_df

    def get_amazon_data(self):
        with open(self.raw_data_path, 'r') as fin:
            #确保字段相同
            resultList = []
            for line in fin:
                try:
                    tempDic = eval(line)
                    resultDic = {}
                    resultDic["user_id"] = tempDic["reviewerID"]
                    resultDic["item_id"] = tempDic["asin"]
                    resultDic["time_stamp"] = tempDic["unixReviewTime"]

                    resultList.append(resultDic)
                except Exception as e:
                    self.logger.info("Error！！！！！！！！！！！！")
                    self.logger.info(e)

            reviews_Electronics_df = pd.DataFrame(resultList)

        with open(self.raw_data_path_meta, 'r') as fin:
            resultList = []
            for line in fin:
                tempDic = eval(line)
                resultDic = {}
                resultDic["cat_id"] = tempDic["categories"][-1][-1]
                resultDic["item_id"] = tempDic["asin"]

                resultList.append(resultDic)

            meta_df = pd.DataFrame(resultList)

        reviews_Electronics_df = pd.merge(reviews_Electronics_df, meta_df, on="item_id")
        print(reviews_Electronics_df.shape)

        resultItemList = []

        def GetMostItemFrency(x):
            if x.shape[0] >= 30:
                resultDic = {}
                resultDic["item_id"] = x["item_id"].tolist()[0]
                resultItemList.append(resultDic)

        reviews_Electronics_df.groupby(["item_id"]).apply(lambda x: GetMostItemFrency(x))
        resultItemList = pd.DataFrame(resultItemList)
        print(resultItemList)

        resultUserList = []

        def GetUserItemFrency(x):
            if x.shape[0] >= 15:
                resultDic = {}
                resultDic["user_id"] = x["user_id"].tolist()[0]
                resultUserList.append(resultDic)

        reviews_Electronics_df.groupby(["user_id"]).apply(lambda x: GetUserItemFrency(x))
        resultUserList = pd.DataFrame(resultUserList)
        print(resultUserList)

        reviews_Electronics_df = pd.merge(reviews_Electronics_df, resultItemList, on="item_id")
        reviews_Electronics_df = pd.merge(reviews_Electronics_df, resultUserList, on="user_id")

        print(reviews_Electronics_df.shape)
        reviews_Electronics_df.to_csv(self.filePath, index=False, encoding="UTF8")
        return reviews_Electronics_df

    #进行两者统计
    def getDataStatistics(self):

        reviews_df = self.origin_data
        # reviews_df = reviews_df.sample(10000)
        user = set(reviews_df["user_id"].tolist())
        print("The user count is " + str(len(user)))
        item = set(reviews_df["item_id"].tolist())
        print("The item count is " + str(len(item)))
        category = set(reviews_df["cat_id"].tolist())
        print("The category  count is " + str(len(category)))

        behavior = reviews_df.shape[0]
        print("The behavior count is " + str(behavior))

        if self.type == "Tmall":
            print(reviews_df)
            behavior_type = set(reviews_df["action_type"].tolist())
            print("The behavior_type count is " + str(len(behavior_type)))
            #to find buy count in different time


        behavior_per_user = reviews_df.groupby(by=["user_id"], as_index=False)["item_id"].count()
        behavior_per_user = behavior_per_user["item_id"].mean()
        print("The avg behavior of each user count is " + str(behavior_per_user))

        behavior_per_item = reviews_df.groupby(by=["item_id"], as_index=False)["user_id"].count()
        behavior_per_item = behavior_per_item["user_id"].mean()
        print("The avg behavior of each item count is " + str(behavior_per_item))

    def top_user_purchase(self):
        result_list = []

        def get_top_item(x):
            print(x)

        self.origin_data.groupby(by=["user_id","item_id"], as_index=False)\
            .apply(lambda x:get_top_item(x))


if __name__ == "__main__":
    get_data_ins = Get_origin_data(type = "Tianchi")
    origin_data = get_data_ins.origin_data
    print(origin_data)




