# This script train and test a deep CNN on ICML 2013 Face recognition database, 
# using entropyloss / multiclasshingeloss / crammerloss / leeloss(not finished)

# To use it, please download and dezip 
# the dataset from this page (fer2013.tar) : 
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import time 
import pandas as pd

import tensorflow as tf
import numpy as np

#choose parameters
LOSS=1 # 0:entropyloss  1:multiclasshingeloss 2:multiclasscrammerloss(sous contrainte b=0) 3:multiclassleeloss(sum(b)=0??)
BATCH_SIZE=10
ITERATION=40
LAMB=0 #devant le terme de régularization
LEARNING_RATE=1e-4
NB_TRAIN=28709
FLAGS = None

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
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, BATCH_SIZE) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
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
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    ## Read the dataset
  df=pd.read_csv('fer2013.csv')
  a=df.as_matrix(['pixels'])
  c=[]

  for i in range(0,len(a)):
    b=np.fromstring(a[i,0],dtype=int,sep=' ')
    c.append(b)
  d=np.asarray(c)

  p=pd.get_dummies(df['emotion'])
  l=p.as_matrix(columns=[p.columns[:]])


   ## neural network implementation
  x=tf.placeholder(tf.float32,[None,2304])

  x_image=tf.reshape(x,[-1,48,48,1])

  #first layer
  W_conv1=weight_variable([5,5,1,32])
  b_conv1=bias_variable([32])

  h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
  h_pool1=max_pool_2x2(h_conv1)

  #second layer
  W_conv2=weight_variable([5,5,32,64])
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
  W_fc2=weight_variable([1024,7])
  b_fc2=bias_variable([7])

  y_ = tf.placeholder(tf.float32, [None, 7])

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

  X,Y=d,l #int(RATIO_TRAIN*len(X))
  X_train,X_test= X[:NB_TRAIN,:] , X[(NB_TRAIN+1):(NB_TRAIN+800),:]
  Y_train,Y_test= Y[:NB_TRAIN,:] , Y[(NB_TRAIN+1):(NB_TRAIN+800),:]

  tps1 = time.clock() 

  #TRAIN
  for k in range(ITERATION):
    #b=random.sample(np.arange(a).astype(int).tolist(),BATCH_SIZE)  #BATCH_SIZE) #random batchs
    batch=np.arange(k*BATCH_SIZE,(k+1)*BATCH_SIZE)%NB_TRAIN     #batchs successifs
    batch_xs, batch_ys = X_train[batch,:] , Y_train[batch,:]
    if k%100 == 0:
       train_accuracy = accuracy.eval(feed_dict={ x:batch_xs, y_:batch_ys, keep_prob: 1.0})
       print("step %d, training accuracy %g"%(k, train_accuracy))
       #print("step %d, learning_rate %g, training accuracy %g"%(k, learning_rate.eval(), train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1})
    
  tps2 = time.clock() 
  print("weights = ")
  print(sess.run(W_fc2))
  print("biais = ")
  print(sess.run(b_fc2))
  print("sum weights = ",sess.run(tf.reduce_sum(W_fc2)))
  print("sum biais = ",sess.run(tf.reduce_sum(b_fc2)))
  print("temps training = ",int(tps2 - tps1)," secondes" )

  # TEST
  print("risque réel = ",sess.run(accuracy, feed_dict={x: X_test,
                                      y_: Y_test,keep_prob: 1}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
tf.app.run()