# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 10:02
# @Author  : zxl
# @FileName: Gated_Behavior_embedding_time_aware_attention.py


import tensorflow.compat.v1 as tf
from util.read_embedding_dic import embedding_csv_dic
import numpy as np
import copy
from Embedding.base_embedding import Base_embedding


class Behavior_embedding_time_aware_attention(Base_embedding):

    def __init__(self,is_training = True, user_count =0 ,
                 item_count=0, category_count=0, max_length_seq=0):
        self.user_count = user_count
        self.item_count = item_count
        self.category_count = category_count
        self.position_count = max_length_seq
        super(Behavior_embedding_time_aware_attention, self).__init__(is_training)  # 此处修改了

        #self.init_placeholders()


    def init_placeholders(self):

        with tf.variable_scope("input_layer"):

            # [B] user id
            self.user_id = tf.placeholder(tf.int32, [None, ],name = "user")
            # [B] item list (user history)
            self.item_list = tf.placeholder(tf.int32, [None,None],name = "item_seq")
            # category list
            self.category_list = tf.placeholder(tf.int32, [None, None],name='category_list')
            # time_list
            self.time_list = tf.placeholder(tf.float32, [None,None], name='time_list')
            # time_last list (the interval between the current item and its last item)
            self.timelast_list = tf.placeholder(tf.float32, [None, self.position_count],name='timelast_list')
            # time_now_list (the interval between the current item and the target item)
            self.timenow_list = tf.placeholder(tf.float32, [None,None], name='timenow_list')
            # position list
            self.position_list = tf.placeholder(tf.int32, [None, None],name='position_list')
            # target item id
            self.target_item_id = tf.placeholder(tf.int32, [None], name='target_item_id')
            # target item id
            self.target_item_category = tf.placeholder(tf.int32, [None], name='target_item_category')
            # target item id
            self.target_item_time = tf.placeholder(tf.float32, [None], name='target_item_time')
            # length of item list
            self.seq_length = tf.placeholder(tf.int32, [None,],name = "seq_length")


            self.adj_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None],name='adj_in')
            self.adj_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None],name='adj_out')

            # self.t_adj_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='t_adj_in')
            # self.t_adj_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='t_adj_out')
            #
            # self.mask_adj_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='mask_adj_in')
            # self.mask_adj_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='mask_adj_out')

            # self.eid_adj_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='eid_adj_in')
            # self.eid_adj_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='eid_adj_out')



    def get_reconsume_embedding(self,reconsume_list,is_reconsume):

        reconsume_lst_emb = tf.nn.embedding_lookup(self.reconsume_emb_lookup_table, reconsume_list)

        target_reconsume_emb = tf.nn.embedding_lookup(self.reconsume_emb_lookup_table, is_reconsume)

        return reconsume_lst_emb,target_reconsume_emb



    def get_adj_matrix(self):

        # edge embedding
        # self.edge_emb_lookup_table = self.init_embedding_lookup_table(name='edge', total_count=self.position_count + 3,
        #                                                               embedding_dim=32,
        #                                                               is_training=self.is_training)
        #
        # in_edge_embedding = tf.nn.embedding_lookup(self.edge_emb_lookup_table,tf.cast(self.eid_adj_in,tf.int32))
        # out_edge_embedding = tf.nn.embedding_lookup(self.edge_emb_lookup_table,tf.cast(self.eid_adj_out,tf.int32))

        return self.adj_in, self.adj_out

    def get_embedding(self,num_units):

        self.reconsume_emb_lookup_table = self.init_embedding_lookup_table(name="reconsume", total_count=2,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)



        # user embedding
        self.user_emb_lookup_table = self.init_embedding_lookup_table(name="user", total_count=self.user_count+3,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)
        #tf.summary.histogram('user_emb_lookup_table', self.user_emb_lookup_table)
        user_embedding = tf.nn.embedding_lookup(self.user_emb_lookup_table, self.user_id)

        # item embedding
        self.item_emb_lookup_table = self.init_embedding_lookup_table(name="item", total_count=self.item_count+3,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)
        #tf.summary.histogram('item_emb_lookup_table', self.item_emb_lookup_table)
        item_list_embedding = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_list)

        # category embedding
        self.category_emb_lookup_table = self.init_embedding_lookup_table(name="category", total_count=self.category_count+3,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)
        #tf.summary.histogram('category_emb_lookup_table', self.category_emb_lookup_table)
        category_list_embedding = tf.nn.embedding_lookup(self.category_emb_lookup_table, self.category_list)

        # position embedding
        self.position_emb_lookup_table = self.init_embedding_lookup_table(name="position",
                                                                          total_count=self.position_count+3,
                                                                          embedding_dim=num_units,
                                                                          is_training=self.is_training)
        #tf.summary.histogram('position_emb_lookup_table', self.position_emb_lookup_table)
        position_list_embedding = tf.nn.embedding_lookup(self.position_emb_lookup_table,
                                                         self.position_list)

        with tf.variable_scope("position_embedding"):

            # behavior_list_embedding = tf.concat([item_list_embedding, category_list_embedding],
            #                                       axis=2,
            #                                       name="seq_embedding_concat")
            # behavior_list_embedding_dense = tf.layers.dense(behavior_list_embedding, num_units,
            #                                                   activation=tf.nn.relu, use_bias=False,
            #                                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            #                                                   name='dense4emb')
            # behavior_list_embedding_dense = behavior_list_embedding_dense+position_list_embedding
            # TODO 只用item list embedding

            behavior_list_embedding_dense = item_list_embedding


        return  user_embedding, \
                behavior_list_embedding_dense,\
                item_list_embedding,\
                category_list_embedding, \
                position_list_embedding,\
                self.time_list,\
                self.timelast_list,\
                self.timenow_list,\
                [self.target_item_id, self.target_item_category,self.target_item_time], \
                self.seq_length,\
                self.item_list



    def tranform_list_ndarray(self,deal_data,max_len,index):

        result = np.zeros([len(self.batch_data),max_len],np.float)

        k = 0
        for t in deal_data:
            for l in range(len(t[1])):
                result[k][l] = t[k][l]
            k += 1

        return result


    def concat_time_emb(self,item_seq_emb,):

        if self.config['concat_time_emb'] == True:
            t_emb = tf.one_hot(self.hist_t, 12, dtype=tf.float32)
            item_seq_emb = tf.concat([item_seq_emb, t_emb], -1)
            item_seq_emb = tf.layers.dense(item_seq_emb, self.config['hidden_units'])
        else:
            t_emb = tf.layers.dense(tf.expand_dims(self.hist_t, -1),
                                    self.config['hidden_units'],
                                    activation=tf.nn.tanh)
            item_seq_emb += t_emb

        return item_seq_emb


    def make_feed_dic_new(self,batch_data):
        user_id = []
        item_list = []
        category_list = []
        time_list = []
        timelast_list = []
        timenow_list = []
        position_list = []
        target_id =[]
        target_category = []
        target_time = []
        length = []

        adj_in = []
        adj_out = []

        t_adj_in = []
        t_adj_out = []

        mask_adj_in = []
        mask_adj_out = []

        eid_adj_in = []
        eid_adj_out = []

        feed_dict = {}
        def normalize_time(time):
            time = np.log(time+np.ones_like(time))
            #_range = np.max(time) - np.min(time)
            return time/(np.mean(time)+1)
            #return (time - np.min(time)) / _range


        for example in batch_data:
            padding_size = [0,int(self.position_count-example[8])]
            user_id.append(example[0])
            item_list.append(np.pad(example[1],padding_size,'constant'))
            category_list.append(np.pad(example[2],padding_size,'constant'))
            time_list.append(np.pad(example[3],padding_size,'constant'))
            timelast_list.append(np.pad(example[4],padding_size,'constant'))
            timenow_list.append(np.pad(example[5],padding_size,'constant'))
            position_list.append(np.pad(example[6],padding_size,'constant'))
            target_id.append(example[7][0])
            target_category.append(example[7][1])
            target_time.append(example[7][2])
            length.append(example[8])

            matrix_padding_size = ((0,self.position_count - example[8]),(0,self.position_count-example[8]))
            matrix_padding_value = ((0,0))
            # u_A_in, u_A_out, in_edge_ids,out_edge_ids,  masks,edge_ids,avg_time = example[11:]
            u_A_in, u_A_out = example[-2:]

            adj_in.append(np.pad(u_A_in,matrix_padding_size,'constant',constant_values=(0,0)))
            adj_out.append(np.pad(u_A_out,matrix_padding_size,'constant',constant_values=(0,0)))

            # t_adj_in.append(np.pad(t_u_A_in,matrix_padding_size,'constant',constant_values=(0,0)))
            # t_adj_out.append(np.pad(t_u_A_out,matrix_padding_size,'constant',constant_values=(0,0)))
            #
            # mask_adj_in.append(np.pad(in_masks,matrix_padding_size,'constant',constant_values=(0,0)))
            # mask_adj_out.append(np.pad(out_masks,matrix_padding_size,'constant',constant_values=(0,0)))

            # eid_adj_in.append(np.pad(in_edge_ids,matrix_padding_size,'constant',constant_values=(0,0)))
            # eid_adj_out.append(np.pad(out_edge_ids,matrix_padding_size,'constant',constant_values=(0,0)))

        feed_dict[self.user_id] = user_id
        feed_dict[self.item_list] = item_list
        feed_dict[self.category_list] = category_list
        feed_dict[self.time_list] = time_list
        feed_dict[self.timelast_list] = timelast_list
        feed_dict[self.timenow_list] = timenow_list
        feed_dict[self.position_list] = position_list
        feed_dict[self.target_item_id] = target_id
        feed_dict[self.target_item_category] = target_category
        feed_dict[self.target_item_time] = target_time
        feed_dict[self.seq_length] = length

        feed_dict[self.adj_in] = adj_in
        feed_dict[self.adj_out] = adj_out

        # feed_dict[self.t_adj_in] = t_adj_in
        # feed_dict[self.t_adj_out] = t_adj_out
        #
        # feed_dict[self.mask_adj_in] = mask_adj_in
        # feed_dict[self.mask_adj_out] = mask_adj_out

        # feed_dict[self.eid_adj_in] = eid_adj_in
        # feed_dict[self.eid_adj_out] = eid_adj_out


        return feed_dict


        #feed_dic[self.]








