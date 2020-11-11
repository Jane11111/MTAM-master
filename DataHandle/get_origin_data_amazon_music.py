from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import datetime
import time
np.random.seed(1234)


class Get_amazon_data_music(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_amazon_data_music, self).__init__(FLAGS = FLAGS)

        self.raw_data_path = "data/raw_data/amazon_cds/CDs_and_Vinyl.json"
        self.raw_data_path_meta = "data/raw_data/amazon_cds/meta_CDs_and_Vinyl.json"

        self.data_path = "data/orgin_data/music.csv"

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
            error_n =0
            for line in fin:
                try:
                    tempDic = eval(line)
                    if "brand" not in tempDic.keys():
                        tempDic["brand"] = "unknown"
                    resultDic = {}
                    resultDic["cat_id"] = tempDic["category"][-1]
                    resultDic["item_id"] = tempDic["asin"]

                    resultList.append(resultDic)
                except Exception as e:
                    error_n+=1
            self.logger.info("total error entries of meta" + str(error_n))

            meta_df = pd.DataFrame(resultList)

        #user_filter = reviews_Electronics_df.groupby("user_id").count()
        #userfiltered = user_filter.sample(frac=0.22)
        #reviews_Electronics_df = reviews_Electronics_df[reviews_Electronics_df['user_id'].isin(userfiltered.index)]
        #print(reviews_Electronics_df.shape)



        reviews_Electronics_df = pd.merge(reviews_Electronics_df, meta_df, on="item_id")
        self.logger.info(reviews_Electronics_df.shape)

        reviews_Electronics_df = self.filter(reviews_Electronics_df)


        reviews_Electronics_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.logger.info(reviews_Electronics_df.shape)
        self.origin_data =  reviews_Electronics_df
        self.logger.info("well done!!")



if __name__ == "__main__":

    get_data_ins = Get_amazon_data(type = "Tianchi")
    print("hello")