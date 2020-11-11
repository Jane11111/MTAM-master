import random
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
import os
import copy
import random
from util.model_log import create_log
from Prepare.mask_data_process import mask_data_process
np.random.seed(1234)
import tensorflow.compat.v1 as tf
class prepare_data_base():
    """
    Get_train_test(type,origin_data,experiment_type)
    generate training set and testing set
    -----------
    :parameter
        type: "Tmall", "Amazon"
        origin_data: Get_origin_data.origin_data
        experiment_type: "BSBE", "lSTSBP", "Atrank"....
        gapnum: number of gaps, default = 6
        user_count_limit: the limit of the data set
        test_frac： train test radio
    """

    # def construct_graph(self,u_input,t_input ):
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_out = np.zeros((position_count, position_count))
    #     u_A_in = np.zeros((position_count, position_count))
    #
    #     # t_u_A_out = np.zeros((position_count, position_count))
    #     # t_u_A_in = np.zeros((position_count, position_count))
    #     #
    #     # in_masks = np.zeros((position_count, position_count))
    #     # out_masks = np.zeros((position_count,position_count))
    #     masks = np.zeros((position_count,position_count))
    #     in_edge_ids = np.zeros((position_count, position_count))
    #     out_edge_ids = np.zeros((position_count, position_count))
    #     edge_ids = np.zeros((position_count,position_count))
    #
    #     sum_time = np.zeros((position_count,))
    #     count_neighbor = np.zeros((position_count,))
    #     for i in np.arange(len(u_input) - 2):
    #         u_lst = np.where(u_input == u_input[i])[0]
    #         v_lst = np.where(u_input == u_input[i + 1])[0]
    #         origin_t = t_input[i+1]
    #         t = np.exp(-t_input[i + 1])  # TODO 这个时间怎么处理
    #
    #         # masks[u_lst[0]][v_lst[0]] = 1.
    #         for u in u_lst:
    #             sum_time[u] += origin_t
    #             count_neighbor[u] += 1
    #             masks[u][v_lst[0]]=1. # 每个邻居只记一次
    #
    #             masks[u][u] = 1. # 包括自己
    #             edge_ids[u][u] = self.FLAGS.max_length_seq+1
    #             edge_ids[u][v_lst[0]]=i+1
    #
    #             for v in v_lst:
    #                 u_A_out[u][v] += 1 / len(v_lst)
    #                 # t_u_A_out[u][v] = t
    #                 # out_masks[u][v] = 1
    #                 out_edge_ids[u][v] = i+1
    #                 u_A_in[v][u] += 1 / len(u_lst)
    #                 # t_u_A_in[v][u] = t
    #                 # in_masks[v][u] = 1
    #                 in_edge_ids[v][u] = i+1
    #
    #         for v in v_lst:
    #             sum_time[v] += origin_t
    #             count_neighbor[v] += 1
    #             masks[v][u_lst[0]] = 1.
    #             masks[v][v] = 1.
    #             edge_ids[v][v] = self.FLAGS.max_length_seq + 1
    #             edge_ids[v][u_lst[0]] = i + 1
    #
    #
    #
    #     u_sum_in = np.reshape(np.sum(u_A_in, 1), (-1, 1))
    #     u_sum_in[np.where(u_sum_in == 0)] = 1
    #     u_A_in = np.divide(u_A_in, u_sum_in)
    #
    #     u_sum_out = np.reshape(np.sum(u_A_out, 1), (-1, 1))
    #     u_sum_out[np.where(u_sum_out == 0)] = 1
    #     u_A_out = np.divide(u_A_out, u_sum_out)
    #
    #     # t_u_sum_in = np.reshape(np.sum(t_u_A_in, 1), (-1, 1))
    #     # t_u_sum_in[np.where(t_u_sum_in == 0)] = 1
    #     # t_u_A_in = np.divide(t_u_A_in, t_u_sum_in)
    #     #
    #     # t_u_sum_out = np.reshape(np.sum(t_u_A_out, 1), (-1, 1))
    #     # t_u_sum_out[np.where(t_u_sum_out == 0)] = 1
    #     # t_u_A_out = np.divide(t_u_A_out, t_u_sum_out)
    #
    #     count_neighbor[np.where(count_neighbor == 0)] = 1
    #     avg_time = sum_time/count_neighbor
    #
    #     return u_A_in, u_A_out,in_edge_ids,out_edge_ids,masks,edge_ids,avg_time

    def construct_graph(self,u_input, t_input):
        position_count = len(u_input)
        u_input = np.array(u_input)
        u_A_out = np.zeros((position_count, position_count))
        u_A_in = np.zeros((position_count, position_count))

        last_item_idxs = np.where(u_input == u_input[-2])[0]
        has_loop ={}
        for i in np.arange(len(u_input) - 2):
            u_lst = np.where(u_input == u_input[i])[0]
            v_lst = np.where(u_input == u_input[i + 1])[0]


            for u in u_lst:
                u_A_out[u][v_lst[0]] += 1  # 每个结点只计算一次
            for v in v_lst:
                u_A_in[v ][u_lst[0]] += 1
            for u in u_lst:  # 自环
                u_A_out[u][u_lst[0]] += 1
                u_A_in[u][u_lst[0]] += 1


        # print(u_A_out)
        u_sum_in = np.reshape(np.sum(u_A_in, 1), (-1, 1))
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A_in, u_sum_in)

        u_sum_out = np.reshape(np.sum(u_A_out, 1), (-1, 1))
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A_out, u_sum_out)

        return u_A_in, u_A_out


    def process_graph_matrix(self,example ):
        item_list = example[1]
        time_list = example[3]

        # u_A_in, u_A_out,  \
        # in_edge_ids, out_edge_ids, \
        # masks,edge_ids, avg_time = self.construct_graph(item_list,time_list )

        u_A_in, u_A_out = self.construct_graph(item_list, time_list)

        example.extend([u_A_in, u_A_out ])
        return example

    def construct_graph_FGNN(self, u_input, t_input):
        position_count = len(u_input)
        u_input = np.array(u_input)
        u_A_out = np.zeros((position_count, position_count))
        u_A_in = np.zeros((position_count, position_count))

        has_loop = {}
        l = len(u_input) - 1

        for i in np.arange(len(u_input) - 2):
            u_lst = np.where(u_input == u_input[i])[0]
            v_lst = np.where(u_input == u_input[i + 1])[0]
            if u_input[i] == u_input[i + 1]:
                has_loop[u_input[i]] = True
            # masks[u_lst[0]][v_lst[0]] = 1.
            for u in u_lst:
                u_A_out[u][v_lst[0]] += 1 / l
            for v in v_lst:
                u_A_in[v][u_lst[0]] += 1 / l
        for i in np.arange(len(u_input - 1)):
            if u_input[i] in has_loop:
                continue
            u_lst = np.where(u_input == u_input[i])[0]
            for u in u_lst:
                u_A_out[u][u_lst[0]] += 1 / l
                u_A_in[u][u_lst[0]] += 1 / l

        # print(u_A_out)
        # u_sum_in = np.reshape(np.sum(u_A_in, 1), (-1, 1))
        # u_sum_in[np.where(u_sum_in == 0)] = 1
        # u_A_in = np.divide(u_A_in, u_sum_in)
        #
        # u_sum_out = np.reshape(np.sum(u_A_out, 1), (-1, 1))
        # u_sum_out[np.where(u_sum_out == 0)] = 1
        # u_A_out = np.divide(u_A_out, u_sum_out)

        return u_A_in, u_A_out
    def process_graph_matrix_FGNN(self,example ):
        item_list = example[1]
        time_list = example[3]

        # u_A_in, u_A_out,  \
        # in_edge_ids, out_edge_ids, \
        # masks,edge_ids, avg_time = self.construct_graph(item_list,time_list )

        u_A_in, u_A_out = self.construct_graph_FGNN(item_list, time_list)

        example.extend([u_A_in, u_A_out ])
        return example

    def __init__(self, FLAGS, origin_data):

        self.FLAGS = FLAGS
        self.length = []
        self.type = FLAGS.type
        self.user_count_limit = FLAGS.user_count_limit
        self.test_frac = FLAGS.test_frac
        self.experiment_type = FLAGS.experiment_type
        self.neg_sample_ratio = FLAGS.neg_sample_ratio
        self.origin_data = origin_data


        self.data_type_error = 0
        self.data_too_short = 0

        # give the random  target value
        #self.target_random_value

        # make origin data dir
        self.dataset_path = 'data/training_testing_data/' + self.type + "_" + \
                                  self.FLAGS.pos_embedding + "_" +      \
                                  self.FLAGS.experiment_data_type+'_' + \
                                  self.FLAGS.causality

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)


        self.dataset_class_pkl = os.path.join(self.dataset_path,'parameters.pkl')
        self.dataset_class_train = os.path.join(self.dataset_path,'train_data.txt')
        self.dataset_class_test = os.path.join(self.dataset_path,'test_data.txt')
        self.dataset_class_dev = os.path.join(self.dataset_path, 'dev_data.txt')
        self.mask_rate = self.FLAGS.mask_rate

        log_ins = create_log()
        self.logger = log_ins.logger

        # init or load
        if FLAGS.init_train_data == True:
            self.origin_data = origin_data
            #Init index for items, users and categories
            self.map_process()


        # load data
        else:
            # load train data
            with open(self.dataset_class_train, 'r') as f:
                self.train_set = []
                L = f.readlines()
                for line in L:
                    line = list(eval(line))

                    if FLAGS.experiment_type == 'FGNN':
                        line = self.process_graph_matrix_FGNN(line )
                    else:
                        line = self.process_graph_matrix(line)

                    self.train_set.append(line)
                    if len(self.train_set) > 500 :
                        break


            # load test data
            with open(self.dataset_class_test, 'r') as f:
                self.test_set = []
                L = f.readlines()
                for line in L:
                    line = list(eval(line))

                    if FLAGS.experiment_type == 'FGNN':
                        line = self.process_graph_matrix_FGNN(line )
                    else:
                        line = self.process_graph_matrix(line )

                    self.test_set.append(line)

                    # if len(self.test_set) >= 20000:
                    #     break
                    if len(self.test_set) >= 200 :
                        break
                        # load test data
            with open(self.dataset_class_dev, 'r') as f:
                self.dev_set = []
                L = f.readlines()
                for line in L:
                    line = list(eval(line))

                    if FLAGS.experiment_type == 'FGNN':
                        line = self.process_graph_matrix_FGNN(line)
                    else:
                        line = self.process_graph_matrix(line)

                    self.dev_set.append(line)

                    # if len(self.dev_set) >= 20000:
                    #     break
                    if len(self.dev_set) >= 200:
                        break

            with open(self.dataset_class_pkl, 'rb') as f:

                data_dic = pickle.load(f)
                self.item_count = data_dic["item_count"]
                self.user_count = data_dic["user_count"]
                self.category_count = data_dic["category_count"]
                #self.gap = data_dic["gap"]
                self.item_category_dic = data_dic["item_category"]
                self.logger.info("load data finish")
                self.logger.info('Size of training set is ' + str(len(self.train_set)))
                self.logger.info('Size of testing set is ' + str(len(self.test_set)))
                self.logger.info('Size of dev set is ' + str(len(self.dev_set)))
                del data_dic

        self.init_train_data = FLAGS.init_train_data

    #give the index of item and category
    def map_process(self):
        """
        Map origin_data to one-hot-coding except time.

        """
        item_le = preprocessing.LabelEncoder()
        user_le = preprocessing.LabelEncoder()
        cat_le = preprocessing.LabelEncoder()

        # get item id list
        item_id = item_le.fit_transform(self.origin_data["item_id"].tolist())
        self.item_count = len(set(item_id))

        # get user id list
        user_id = user_le.fit_transform(self.origin_data["user_id"].tolist())
        self.user_count = len(set(user_id))

        # get category id list
        cat_id = cat_le.fit_transform(self.origin_data["cat_id"].tolist())
        self.category_count = len(set(cat_id))

        self.item_category_dic = {}
        for i in range(0, len(item_id)):
            self.item_category_dic[item_id[i]] = cat_id[i]

        self.logger.warning("item Count :" + str(len(item_le.classes_)))
        self.logger.info("user count is " + str(len(user_le.classes_)))
        self.logger.info("category count is " + str(len(cat_le.classes_)))

        # _key:key的列表，_map:key的列表加编号
        self.origin_data['item_id'] = item_id
        self.origin_data['user_id'] = user_id
        self.origin_data['cat_id'] = cat_id

        # 根据reviewerID、unixReviewTime编号进行排序（sort_values：排序函数）
        self.origin_data = self.origin_data.sort_values(['user_id', 'time_stamp'])

        # 重新建立索引
        self.origin_data = self.origin_data.reset_index(drop=True)
        return self.user_count, self.item_count

    #choose one for the action which are too close
    def filter_repetition(self):
        pass


    def get_train_test(self):
        """
        Generate training set and testing set with the mask_rate.
        The training set will be stored in training_set.pkl.
        The testing set will be stored in testing_set.pkl.
        dataset_path: 'data/training_testing_data/'
        :param
            data_size: number of samples
        :returns
            train_set: (user_id, item_list, (factor1_list, factor2,..., factorn), masked_item, label）
            test_set: (user_id, item_list, (factor1, factor2,..., factorn), (masked_item_positive,masked_item_negtive)）
            e.g. Amazon_bsbe
            train_set: (user_id, item_list, (time_interval_list, category_list), masked_item, label）
            test_set: (user_id, item_list,(time_interval_list, category_list), (masked_item_positive,masked_item_negtive)）
            e.g. Amazon_bsbe
            train_set: (user_id, item_list, (time_interval_list, category_list, action_list), masked_item, label）
            test_set: (user_id, item_list, (time_interval_list, category_list, action_list), (masked_item_positive,masked_item_negtive)）

        """
        if self.init_train_data == False:
            return self.train_set, self.test_set,self.dev_set

        self.data_set = []
        self.train_set = []
        self.test_set = []
        self.dev_set = []

        self.now_count = 0

        #data_handle_process为各子类都使用的函数
        self.origin_data.groupby(["user_id"]).filter(lambda x: self.data_handle_process(x))
        # self.format_train_test()

        random.shuffle(self.train_set)
        random.shuffle(self.test_set)
        random.shuffle(self.dev_set)

        self.logger.info('Size of training set is ' + str(len(self.train_set)))
        self.logger.info('Size of testing set is ' + str(len(self.test_set)))
        self.logger.info('Size of dev set is ' + str(len(self.dev_set)))
        self.logger.info('Data type error size  is ' + str(self.data_type_error))
        self.logger.info('Data too short size is ' + str(self.data_too_short))


        with open(self.dataset_class_pkl, 'wb') as f:
            data_dic = {}
            data_dic["item_count"] = self.item_count
            data_dic["user_count"] = self.user_count
            data_dic["category_count"] = self.category_count
            #data_dic["gap"] = self.gap
            data_dic["item_category"] = self.item_category_dic
            pickle.dump(data_dic, f, pickle.HIGHEST_PROTOCOL)

        # train text 和 test text 使用文本
        self.save(self.train_set,self.dataset_class_train)
        self.save(self.test_set,self.dataset_class_test)
        self.save(self.dev_set, self.dataset_class_dev)

        return self.train_set, self.test_set, self.dev_set

    def data_handle_process_base(self, x):
        behavior_seq = copy.deepcopy(x)
        if self.FLAGS.remove_duplicate == True:
            behavior_seq = behavior_seq.drop_duplicates(keep="last")

        behavior_seq = behavior_seq.sort_values(by=['time_stamp'], na_position='first')
        behavior_seq = behavior_seq.reset_index(drop=True)
        columns_value = behavior_seq.columns.values.tolist()
        if "user_id" not in columns_value:
            self.data_type_error = self.data_type_error + 1
            return

        pos_list = behavior_seq['item_id'].tolist()  # asin属性的值
        length = len(pos_list)

        #limit length
        #if length < 2:
            #self.data_too_short = self.data_too_short + 1
            #return None

        # if length > self.FLAGS.length_of_user_history:
        #     behavior_seq = behavior_seq.tail(self.FLAGS.length_of_user_history)

        # user limit
        # if self.now_count > self.user_count_limit:
        #     return None

        self.now_count = self.now_count + 1
        # test
        behavior_seq = behavior_seq.reset_index(drop=True)
        return behavior_seq

    #给出基本操作
    def data_handle_process(self, x):
        if np.random.randint(0, 100) < 2:
            test_user=True
        else:
            test_user = False
        #Sort User sequence by time and delete sequences whose lenghts are not in [20,150]
        behavior_seq = self.data_handle_process_base(x)
        if behavior_seq is None:
            return

        mask_data_process_ins = mask_data_process(behavior_seq = behavior_seq)

        mask_data_process_ins.get_mask_index_list_behaivor()
        #根据测试训练的比例 来划分

        for index in mask_data_process_ins.mask_index_list:

            #这里只取单项
            user_id, item_seq_temp, factor_list = \
                mask_data_process_ins.mask_process_unidirectional(self.FLAGS.causality,
                                                                  index=index,
                                                                  time_window=24 * 3600 * 35,
                                                                  lengeth_limit=self.FLAGS.length_of_user_history)


            cat_list = factor_list[0]

            #换算成小时
            time_list = [int(x / 3600) for x in factor_list[1]]
            target_time = int(mask_data_process_ins.time_stamp_seq[index] / 3600)


            #mask the target item value
            item_seq_temp.append(self.item_count + 1)
            #mask the target category value
            cat_list.append(self.category_count + 1)

            #update time
            timelast_list, timenow_list = mask_data_process_ins.pro_time_method(time_list,target_time)
            position_list = mask_data_process_ins.proc_pos_emb(time_list)

            #进行padding的填充,便于对齐
            time_list.append(target_time)
            timelast_list.append(0)
            timenow_list.append(0)
            if index > 49:
                position_list.append(49)
            else:
                position_list.append(index)
            target_id = mask_data_process_ins.item_seq[index]
            target_category = self.item_category_dic[mask_data_process_ins.item_seq[index]]

            #以小时为准
            if   index == len(mask_data_process_ins.mask_index_list):
                self.test_set.append((user_id, item_seq_temp, cat_list, time_list,
                                      timelast_list, timenow_list, position_list,
                                      [target_id, target_category, target_time],
                                      len(item_seq_temp)))
            elif   index == len(mask_data_process_ins.mask_index_list)-1:
                self.dev_set.append((user_id, item_seq_temp, cat_list, time_list,
                                      timelast_list, timenow_list, position_list,
                                      [target_id, target_category, target_time],
                                      len(item_seq_temp)))
            else:
                '''
                if np.random.randint(0,20)<2:
                    self.test_set.append((user_id, item_seq_temp, cat_list, time_list,
                                          timelast_list, timenow_list, position_list,
                                          [target_id, target_category, target_time],
                                          len(item_seq_temp)))
                else:'''

                self.train_set.append((user_id, item_seq_temp, cat_list, time_list,
                                       timelast_list, timenow_list, position_list,
                                       [target_id, target_category, target_time],
                                       len(item_seq_temp)))


    def format_train_test(self):
        pass


    def get_gap_list(self, gapnum):
        gap = []
        for i in range(1, gapnum):
            if i == 1:
                gap.append(60)
            elif i == 2:
                gap.append(60 * 60)
            else:
                gap.append(3600 * 24 * np.power(2, i - 3))

        self.gap = np.array(gap)

    #给出写入文件
    def save(self,data_list,file_path):
        fp = open(file_path, 'w+')
        for i in data_list:
            fp.write(str(i) + '\n')
        fp.close()


