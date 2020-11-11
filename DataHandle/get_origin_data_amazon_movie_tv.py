from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
import traceback
import datetime
import time
np.random.seed(1234)
from config.model_parameter import model_parameter


class Get_amazon_data_movie_tv(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_amazon_data_movie_tv, self).__init__(FLAGS = FLAGS)

        self.raw_data_path = "data/raw_data/MoviesTV/amazon_movie/Movies_and_TV.json"
        self.raw_data_path_meta = "data/raw_data/MoviesTV/amazon_movie/meta_Movies_and_TV.json"

        self.data_path = "data/orgin_data/movie_tv.csv"

        if FLAGS.init_origin_data == True:

            self.get_amazon_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_amazon_data(self):


        with open(self.raw_data_path, 'r') as fin:
            #确保字段相同
            resultList = []
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
                    self.logger.info("Error！！！！！！！！！！！！")
                    self.logger.info(e)
                    traceback.print_exc()

            reviews_Electronics_df = pd.DataFrame(resultList)


        with open(self.raw_data_path_meta, 'r') as fin:
            resultList = []
            for line in fin:
                try:
                    line = line.replace("false", "False")
                    tempDic = eval(line)
                    resultDic = {}

                    if "category" in tempDic.keys() and len(tempDic['category']) > 0:
                        resultDic["cat_id"] = tempDic["category"][-1]
                    else:
                        resultDic["cat_id"] = "none"

                    resultDic["item_id"] = tempDic["asin"]
                    resultList.append(resultDic)

                except Exception as e:
                    self.logger.info("Error！！！！！！！！！！！！")
                    self.logger.info(e)
                    traceback.print_exc()


            meta_df = pd.DataFrame(resultList)

        reviews_beauty_df = pd.merge(reviews_Electronics_df, meta_df, on="item_id")

        #print(reviews_Electronics_df.shape)
        #print(reviews_Electronics_df_filter.shape)

        reviews_beauty_df = self.filter(reviews_beauty_df)


        reviews_beauty_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data =  reviews_beauty_df



if __name__ == "__main__":

    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS

    ins = Get_amazon_data_movie_tv(FLAGS=FLAGS)
    ins.getDataStatistics()


    # print(origin_data)





