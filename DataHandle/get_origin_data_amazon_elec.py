from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import datetime
import time
np.random.seed(1234)


class Get_amazon_data_elec(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_amazon_data_elec, self).__init__(FLAGS = FLAGS)

        self.raw_data_path = "data/raw_data/amazon_electronics/Electronics.json"
        self.raw_data_path_meta = "data/raw_data/amazon_electronics/meta_Electronics.json"

        self.data_path = "data/orgin_data/elec.csv"

        if FLAGS.init_origin_data == True:

            self.get_amazon_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_amazon_data(self):


        with open(self.raw_data_path, 'r') as fin:
            #确保字段相同
            resultList = []
            error_n =0
            for line in fin:
                try:
                    line = line.replace("true","True")
                    line = line.replace("false", "False")
                    tempDic = eval(line)
                    resultDic = {}
                    resultDic["user_id"] = tempDic["reviewerID"]
                    resultDic["item_id"] = tempDic["asin"]
                    resultDic["time_stamp"] = tempDic["unixReviewTime"]

                    resultList.append(resultDic)
                except Exception as e:
                    error_n+=1
                    #self.logger.info("Error！！！！！！！！！！！！")
                    #self.logger.info(e)
            self.logger.info("total error entries"+str(error_n))





            reviews_Electronics_df = pd.DataFrame(resultList)

        with open(self.raw_data_path_meta, 'r') as fin:
            resultList = []
            for line in fin:
                tempDic = eval(line)
                resultDic = {}
                resultDic["cat_id"] = tempDic["category"][-1]
                resultDic["item_id"] = tempDic["asin"]

                resultList.append(resultDic)

            meta_df = pd.DataFrame(resultList)

        # user_filter = reviews_Electronics_df.groupby("user_id").count()
        # userfiltered = user_filter.sample(frac=0.2)
        # reviews_Electronics_df = reviews_Electronics_df[reviews_Electronics_df['user_id'].isin(userfiltered.index)]
        # print(reviews_Electronics_df.shape)



        reviews_Electronics_df = pd.merge(reviews_Electronics_df, meta_df, on="item_id")
        print(reviews_Electronics_df.shape)

        reviews_Electronics_df = self.filter(reviews_Electronics_df)


        reviews_Electronics_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data =  reviews_Electronics_df
        print("well done!!")



if __name__ == "__main__":

    get_data_ins = Get_amazon_data(type = "Tianchi")
    print("hello")


    # print(origin_data)





