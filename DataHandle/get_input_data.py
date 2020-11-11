import numpy as np

class DataInput:

    def __init__(self,data,batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) //self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        data_slice = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,len(self.data))]

        self.i += 1
        #
        # u,i,y,sl = [],[],[],[]
        # for t in data_slice:
        #     u.append(t[0])#user id
        #     i.append(t[3])#post_list[i]未来的行为
        #     y.append(t[4])#0 or 1
        #     sl.append(len(t[1]))#hist_i的长度
        # # max_sl = max(sl)
        # #
        # # hist_i = np.zeros([len(data_slice),max_sl],np.int64)
        # # hist_t = np.zeros([len(data_slice),max_sl],np.float32)
        # #
        # # k = 0
        # # for t in data_slice:
        # #     for l in range(len(t[1])):
        # #         hist_i[k][l] = t[1][l]#200 x len(t[1])
        # #         hist_t[k][l] = t[2][l]
        # #     k += 1

        return self.i,data_slice

class DataInputTest:
    def __init__(self,data,batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,len(self.data))]

        self.i += 1

        u,i,j,sl = [],[],[],[]

        for t in ts:
            u.append(t[0])
            i.append(t[3][0])
            j.append(t[3][1])
            sl.append(len(t[1]))

        max_sl = max(sl)
        #print(max_sl)

        hist_i = np.zeros([len(ts),max_sl],np.int64)
        hist_t = np.zeros([len(ts),max_sl],np.float32)

        k = 0
        #print(ts)
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                #print(k,l,hist_i[k][l])
                hist_t[k][l] = t[2][l]
                #print(hist_t[k][l])

            k += 1
        #print(hist_i)
        return self.i,(u,i,j,hist_t,hist_i,sl)