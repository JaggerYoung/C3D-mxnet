import os,sys
import find_mxnet
import mxnet as mx
import numpy as np
import argparse
import logging
import cv2
import random
import glob

from c3d_symbol import get_symbol

BATCH_SIZE = 10
NUM_SAMPLES = 28

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
         self.data = data
         self.label = label
         self.data_names = data_names
         self.label_names = label_names

         self.pad = 0
         self.index = None

    @property
    def provide_data(self):
         return [(n,x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
         return [(n,x.shape) for n,x in zip(self.label_names, self.label)]

def readData(FileName):
    data_1 = []
    data_2 = []
    data_3 = []
    f = open(FileName,'r')
    total = f.readlines()
    print len(total)
    random.shuffle(total)

    for eachLine in range(len(total)):
        tmp = total[eachLine].split('\n')
        tmp_1,tmp_2 = tmp[0].split(' ',1)
        tmp_1 = '/data/zhigang.yang/UCF-101'+tmp_1
	data_1.append(tmp_1)
        data_2.append(int(tmp_2))
        #data_3.append(tmp_3)
    f.close()
    #print data_1
    return (data_1, data_2)

def ImageSeqToMatrix(dirName, num, data_shape):
    pic = []
    #print dirName
    for filename in glob.glob(dirName+'/*.jpg'):
        pic.append(filename)
    pic.sort()
    #print len(pic)
    ret = []
    len_pic = len(pic)
    tmp = len_pic/num
    for i in range(num):
        ret.append(pic[i * tmp])
    r_1 = []
    g_1 = []
    b_1 = []
    mat = []
    
    for i in range(len(ret)):
        img = cv2.imread(ret[i], cv2.IMREAD_COLOR)
	#img = img.resize(data_shape[2],data_shape[3])
        b,g,r = cv2.split(img)
	r = cv2.resize(r, (data_shape[3], data_shape[2]))
	g = cv2.resize(g, (data_shape[3], data_shape[2]))
	b = cv2.resize(b, (data_shape[3], data_shape[2]))
	r = np.multiply(r, 1/255.0)
	g = np.multiply(g, 1/255.0)
	b = np.multiply(b, 1/255.0)
	r_1.append(r)
	g_1.append(g)
	b_1.append(b)
	#mat.append(img)
    mat.append(r_1)
    mat.append(g_1)
    mat.append(b_1)
    #print len(mat),len(mat[0][0])
    return mat

class UCFIter(mx.io.DataIter):
    def __init__(self, fname, num, batch_size, data_shape):
        self.batch_size = batch_size
        self.fname = fname
        self.data_shape = data_shape
        self.count = num/batch_size
        (self.data_1, self.data_2) = readData(fname)
        
	self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = [('label', (batch_size, ))]
        print len(self.data_1)       
 
    def __iter__(self):
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                idx = k * batch_size + i
                pic = ImageSeqToMatrix(self.data_1[idx], NUM_SAMPLES, self.data_shape)
		data.append(pic)
		label.append(int(self.data_2[idx]))
	    
	    data_all = [mx.nd.array(data)]
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch

    def reset(self):
        pass

def Accuracy(label, pred):
    hit = 0.
    total = 0.
    label = label.T.reshape(-1,1)
    for i in range(BATCH_SIZE):
        maxIdx = np.argmax(pred[i])
	if maxIdx == int(label[i]):
	    hit += 1.0
	total += 1.0
    return hit/total

if __name__ == '__main__':

    train_num = 9537
    test_num = 3783

    #train_num = 1820
    #test_num = 831

    batch_size = BATCH_SIZE
    data_shape = (3, NUM_SAMPLES, 122, 122)
    num_label = 101
        
    devs = [mx.context.gpu(3)]
    network = get_symbol(num_label)    
    
    train_file = '/home/users/zhigang.yang/mxnet/example/C3D-mxnet/data/train.list'
    test_file = '/home/users/zhigang.yang/mxnet/example/C3D-mxnet/data/test.list'

    data_train = UCFIter(train_file, train_num, batch_size, data_shape)
    data_val = UCFIter(test_file, test_num, batch_size, data_shape)

    print data_train.provide_data,data_train.provide_label 
   
    model = mx.model.FeedForward(ctx           = devs,
                                 symbol        = network,
                                 num_epoch     = 100,
                                 learning_rate = 0.003,
                                 momentum      = 0.009,
                                 wd            = 0.005,
                                 initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34))
    
    import logging 
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)    

    batch_end_callbacks = [mx.callback.Speedometer(BATCH_SIZE, 1000)]    
    print 'begin fit'
    
    eval_metrics = ['accuracy']
    model.fit(X = data_train, eval_data = data_val, eval_metric = eval_metrics, batch_end_callback = batch_end_callbacks)       
