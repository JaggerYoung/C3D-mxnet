import find_mxnet
import mxnet as mx  
import numpy as np  
import cv2  
import matplotlib.pyplot as plt  
import logging  

logger = logging.getLogger()  
logger.setLevel(logging.DEBUG)  

input_data = mx.symbol.Variable('data')
label = mx.symbol.Variable('label')

## 1st group 
conv1 = mx.symbol.Convolution(data=input_data, kernel=(3,3,3), stride=(1,1,1), pad=(1,1,1), num_filter=64, name='conv1', cudnn_tune='fastest', layout='NCDHW')
relu1 = mx.symbol.Activation(data=conv1, act_type='relu')
pool1 = mx.symbol.Pooling(data=relu1, pool_type='max', kernel=(1,2,2), stride=(1,2,2))

## 2nd group
conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3,3), stride=(1,1,1), pad=(1,1,1), num_filter=128, name='conv2', cudnn_tune='fastest', layout='NCDHW')
relu2 = mx.symbol.Activation(data=conv2, act_type='relu')
pool2 = mx.symbol.Pooling(data=relu2, pool_type='max', kernel=(2,2,2), stride=(2,2,2))

## 3rd group
conv3a = mx.symbol.Convolution(data=pool2, kernel=(3,3,3), stride=(1,1,1), pad=(1,1,1), num_filter=256, cudnn_tune='fastest', layout='NCDHW')
relu3a = mx.symbol.Activation(data=conv3a, act_type='relu')
#conv3b = mx.symbol.Convolution(data=relu3a, kernel=(3,3,3), stride=(1,1,1), num_filter=256)
#relu3b = mx.symbol.Activation(data=conv3b, act_type='relu')
#pool3b = mx.symbol.Pooling(data=relu3b, pool_type='max', kernel=(2,2,2), stride=(2,2,2))
pool3b = mx.symbol.Pooling(data=relu3a, pool_type='max', kernel=(2,2,2), stride=(2,2,2))

## 4th group
conv4a = mx.symbol.Convolution(data=pool3b, kernel=(3,3,3), stride=(1,1,1), pad=(1,1,1), num_filter=256, cudnn_tune='fastest', layout='NCDHW')
relu4a = mx.symbol.Activation(data=conv4a, act_type='relu')
#conv4b = mx.symbol.Convolution(data=relu4a, kernel=(3,3,3), stride=(1,1,1), num_filter=512)
#relu4b = mx.symbol.Activation(data=conv4b, act_type='relu')
#pool4b = mx.symbol.Pooling(data=relu4b, pool_type='max', kernel=(2,2,2), stride=(2,2,2))
pool4b = mx.symbol.Pooling(data=relu4a, pool_type='max', kernel=(2,2,2), stride=(2,2,2))

## 5th group
conv5a = mx.symbol.Convolution(data=pool4b, kernel=(3,3,3), stride=(1,1,1), pad=(1,1,1), num_filter=256, cudnn_tune='fastest', layout='NCDHW')
relu5a = mx.symbol.Activation(data=conv5a, act_type='relu')
#conv5b = mx.symbol.Convolution(data=relu5a, kernel=(3,3,3), stride=(1,1,1), num_filter=512)
#relu5b = mx.symbol.Activation(data=conv5b, act_type='relu')
#pool5b = mx.symbol.Pooling(data=relu5b, pool_type='max', kernel=(2,2,2), stride=(2,2,2))
pool5b = mx.symbol.Pooling(data=relu5a, pool_type='max', kernel=(2,2,2), stride=(2,2,2))

## 6th group
flatten6 = mx.symbol.Flatten(data=pool5b)
fc6 = mx.symbol.FullyConnected(data=flatten6, num_hidden=2048)
relu6 = mx.symbol.Activation(data=fc6, act_type='relu')
drop6 = mx.symbol.Dropout(data=relu6, p=0.5)
fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=2048)
relu7 = mx.symbol.Activation(data=fc7, act_type='relu')
drop7 = mx.symbol.Dropout(data=relu7, p=0.5)

#Loss
fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=num_classes)
softmax = mx.symbol.SoftmaxOutput(data=fc8, label=label, name='softmax')

#view
mx.viz.plot_network(mlp).view()  
