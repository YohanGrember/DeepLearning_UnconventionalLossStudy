# coding: utf8

# This script train and test a deep CNN on CIFAR10 database, 
# using entropyloss / multiclasshingeloss / crammerloss / leeloss(not finished)

# To use it, please download and dezip 
# the dataset from this page: 
# http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
#import random
import time 
#import pandas as pd
#from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#choose parameters
LOSS=1 # 0:entropyloss  1:multiclasshingeloss 2:multiclasscrammerloss(sous contrainte b=0) 3:multiclassleeloss(sum(b)=0??)
BATCH_SIZE=50
ITERATION=20
LAMB=0 #devant le terme de regularisation
LEARNING_RATE=1e-4
NB_TRAIN=5000
FLAGS = None

 
# Converts a row-major RGB image stored in one row into a standard RGB array
def cifar_data_to_array(batch,i):
    data = batch['data'][i]
    array = np.zeros((32,32,3), dtype=np.uint8)
    for color in xrange(3):
        for i in xrange(32):
                array[i,:,color] = data[32*i + 1024*color:32*(i+1) + 1024*color]
    return array
    
def get_xtrain_and_ytrain(batch1, batch2, batch3, batch4, batch5):
    train_images_list = []
    train_labels_list=[]
    for i in xrange(len(batch1['data'])):
        train_images_list.append(cifar_data_to_array(batch1,i))
        train_labels_list.append(batch1['labels'][i])
    for i in xrange(len(batch2['data'])):
        train_images_list.append(cifar_data_to_array(batch2,i))
        train_labels_list.append(batch1['labels'][i])
    for i in xrange(len(batch3['data'])):
        train_images_list.append(cifar_data_to_array(batch3,i))
        train_labels_list.append(batch1['labels'][i])
    for i in xrange(len(batch4['data'])):
        train_images_list.append(cifar_data_to_array(batch4,i))
        train_labels_list.append(batch1['labels'][i])
    for i in xrange(len(batch5['data'])):
        train_images_list.append(cifar_data_to_array(batch5,i))
        train_labels_list.append(batch1['labels'][i])
    y_train = get_dummies(np.asarray(train_labels_list), 10)    
    return np.asarray(train_images_list), y_train
    
def get_dummies(y, n_labels):
    y_train = np.zeros((len(y),n_labels), dtype=int)
    for i in xrange(len(y)):
        y_train[i][y[i]] = 1
    return y_train
    
def shuffle_xtrain_and_ytrain(X_train, Y_train):
    # Creates a random order for images
    a = np.arange(len(X_train))
    np.random.shuffle(a)
    # Set the images and labels in the same, new order
    return X_train.take(a,0), Y_train.take(a,0)
    
    
    
def get_xtest_and_ytest(batchtest):
    test_images_list = []
    test_labels_list = []
    for i in xrange(len(batchtest['data'])):
        test_images_list.append(cifar_data_to_array(batchtest,i))
        test_labels_list.append(batchtest['labels'][i])
    return np.asarray(test_images_list), np.asarray(test_labels_list)
    
def get_indexes(k, batch_size, len_x_train):
    if k*batch_size%len_x_train < batch_size:
        return 0, (k+1)*batch_size % len_x_train
    elif (k*batch_size) % len_x_train + batch_size > len_x_train:
        return k*batch_size % len_x_train, len_x_train
    else:
        return k*batch_size % len_x_train, (k+1)*batch_size % len_x_train

    
# Function given by http://www.cs.toronto.edu/~kriz/cifar.html to decode batch files
def unpickle(string):
    import cPickle
    fo = open(string, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
def entropyloss(scores,classes):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, classes))

def multiclasshingeloss(scores, classes):
  scores=tf.transpose(scores)
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, BATCH_SIZE) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.nn.relu((1 - true_scores + scores) * (1 - classes))
  l=tf.reduce_sum(L,0)
  final_loss=tf.reduce_mean(l)
  return final_loss

def normalization(scores):
  return scores-tf.tile(tf.transpose([tf.reduce_mean(scores,1)]),[1,tf.cast(scores.get_shape()[1], dtype=tf.int32)])

def multiclasscrammerloss(scores, classes):
  scores=tf.transpose(scores)
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, BATCH_SIZE) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L=tf.nn.relu(1-true_scores + tf.reduce_max(scores*(1-classes),reduction_indices=[0]))
  final_loss=tf.reduce_mean(L)
  return final_loss

def multiclassleeloss(scores,classes,nb_class):
  scores=tf.transpose(scores)
  classes=tf.transpose(classes)
#  true_classes = tf.argmax(classes, 0)
#  idx_flattened = tf.range(0, BATCH_SIZE) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
#  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
#                          idx_flattened)
  L = tf.nn.relu( (1/(nb_class-1) + scores) * (1 - classes))
  l=tf.reduce_sum(L,0)
  final_loss=tf.reduce_mean(l)
  return final_loss

def regularization_loss(lamb,weights):
  return lamb*tf.reduce_sum(tf.square(weights)) 

def weight_variable(shape):
  initial= tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1,shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
  
    
def main(_):


    
    tps1 = time.clock() 
    
    batch_1 = unpickle("cifar-10-batches-py/data_batch_1")
    batch_2 = unpickle("cifar-10-batches-py/data_batch_2")
    batch_3 = unpickle("cifar-10-batches-py/data_batch_3")
    batch_4 = unpickle("cifar-10-batches-py/data_batch_4")
    batch_5 = unpickle("cifar-10-batches-py/data_batch_5")
    batch_test = unpickle("cifar-10-batches-py/test_batch")

    X_train, Y_train = get_xtrain_and_ytrain(batch_1, batch_2, batch_3, batch_4, batch_5)
    X_test, Y_test = get_xtest_and_ytest(batch_test)

    tps2 = time.clock() 
    print("Time elapsed during data import and images reshaping: %g secondes" %(tps2 - tps1))

    ## neural network implementation
#    x=tf.placeholder(tf.float32,[None,3072])
    x=tf.placeholder(tf.float32,[None,32,32,3])
#    x_image=tf.reshape(x,[-1,32,32,3])
    
    #first layer red
    W_conv1=weight_variable([5,5,3,64])
    b_conv1=bias_variable([64])
    
    h_conv1=tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
    h_pool1=max_pool_2x2(h_conv1)  

#    #first layer green
#    W_conv11=weight_variable([5,5,3,32])
#    b_conv11=bias_variable([32])
#    
#    h_conv11=tf.nn.relu(conv2d(x,W_conv11)+b_conv11)
#    h_pool11=max_pool_2x2(h_conv11)  
#    
#    #first layer blue
#    W_conv11=weight_variable([5,5,3,32])
#    b_conv11=bias_variable([32])
#    
#    h_conv11=tf.nn.relu(conv2d(x,W_conv11)+b_conv11)
#    h_pool11=max_pool_2x2(h_conv11)  
#    
    
    #second layer
    W_conv2=weight_variable([5,5,64,64])
    b_conv2=bias_variable([64])

    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2=max_pool_2x2(h_conv2)

    #densely connected layer
    W_fc1=weight_variable([12*12*64,1024])
    b_fc1=bias_variable([1024])
    
    h_pool2_flat=tf.reshape(h_pool2,[-1,12*12*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

    #drop out
    keep_prob=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    #readout layer
    W_fc2=weight_variable([1024,10])
    b_fc2=bias_variable([10])

    y_ = tf.placeholder(tf.float32, [None, 10])

    #define the loss and gradient descent
    if LOSS==0:
        y = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
        svm_loss=entropyloss(y,y_) + regularization_loss(LAMB,W_fc2)
    elif LOSS==1:
        y= normalization(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
        svm_loss = multiclasshingeloss(y,y_) + regularization_loss(LAMB,W_fc2)
    elif LOSS==2:
        y = normalization(tf.matmul(h_fc1_drop,W_fc2))
        svm_loss=multiclasscrammerloss(y,y_) + regularization_loss(LAMB,W_fc2)
    else:
        y = normalization(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
        svm_loss=multiclassleeloss(y,y_,7) + regularization_loss(LAMB,W_fc2) 

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(svm_loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()

    tps3 = time.clock() 

    #TRAIN
    for k in range(ITERATION):
        # If idx_min < BATCH_SIZE, idx
        idx_min, idx_max = get_indexes(k, BATCH_SIZE, len(X_train))
        if idx_min == 0:
            X_train, Y_train = shuffle_xtrain_and_ytrain(X_train, Y_train)
            print("starting training of epoch %d" %int(k*BATCH_SIZE/len(X_train)))
        batch_xs, batch_ys = X_train[idx_min:idx_max] , Y_train[idx_min:idx_max]
        if k%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={ x:batch_xs, y_:batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(k, train_accuracy))
            #print("step %d, learning_rate %g, training accuracy %g"%(k, learning_rate.eval(), train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob: .5})
    tps4 = time.clock() 
    
    print("weights = ")
    print(sess.run(W_fc2))
    print("biais = ")
    print(sess.run(b_fc2))
    print("sum weights = ",sess.run(tf.reduce_sum(W_fc2)))
    print("sum biais = ",sess.run(tf.reduce_sum(b_fc2)))
    print("temps training = ",int(tps4 - tps3)," secondes" )

    # TEST
    print("testing accuracy = ",sess.run(accuracy, feed_dict={x: X_test,
                                        y_: Y_test,keep_prob: 1}))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    main(_)
#tf.app.run()