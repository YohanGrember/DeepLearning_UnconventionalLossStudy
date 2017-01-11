# -*- coding: utf-8 -*-

#This script train and test a deep CNN on MNIST database, 
#using entropyloss / multiclasshingeloss / crammerloss / leeloss(not finished)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
#####import random
import time 
# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

#choose parameters
LOSS=0 # 0:entropyloss  1:multiclasshingeloss 2:multiclasscrammerloss(sous contrainte b=0) 3:multiclassleeloss(sum(b)=0??)
BATCH_SIZE=50
ITERATION=3000
LAMBDA=5e-5 #devant le terme de régularisation
FLAGS = None

# Set up the learning rate : decayed_learning_rate = INITIAL_LEARNING_RATE * DECAY_RATE ^ (global_step / DECAY_STEPS)
INITIAL_LEARNING_RATE=1e-4
DECAY_STEPS = 500
DECAY_RATE = 0.85



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
  

#def regularization_loss(lamb,weights):
#  return lamb*tf.reduce_sum(tf.square(weights)) 
  
def regularization_loss(lamb,weights_list):
  loss = 0  
  for weights in weights_list:
    loss += lamb*tf.cast(tf.reduce_sum(tf.square(weights)),tf.float32)
  return loss

def transformData(data): #from [1 3 2 2 1] to [[1 0 0] [0 1 0] ...]
  batch_yss=np.tile(np.transpose([data]),[1,3])
  for i in range(0,len(data)):
    if data[i]==0:
        batch_yss[i]=[1,0,0]
    elif data[i]==1:
        batch_yss[i]=[0,1,0]
    else:
        batch_yss[i]=[0,0,1]
  return batch_yss

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
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  x=tf.placeholder(tf.float32,[None,784])
  x_image=tf.reshape(x,[-1,28,28,1])

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
  W_fc1=weight_variable([7*7*64,1024])
  b_fc1=bias_variable([1024])

  h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
  h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

  #drop out
  keep_prob=tf.placeholder(tf.float32)
  h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

  #readout layer
  W_fc2=weight_variable([1024,10])
  b_fc2=bias_variable([10])

  y_ = tf.placeholder(tf.float32, [None, 10])

  #define the loss and gradient descent
  weights_list = [W_conv1, W_conv2, W_fc1, W_fc2]

  if LOSS==0:
    y = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    svm_loss=entropyloss(y,y_) + regularization_loss(LAMBDA,weights_list)
  elif LOSS==1:
    y= normalization(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
    svm_loss = multiclasshingeloss(y,y_) + regularization_loss(LAMBDA,weights_list)
  elif LOSS==2:
    y = normalization(tf.matmul(h_fc1_drop,W_fc2))
    svm_loss=multiclasscrammerloss(y,y_) + regularization_loss(LAMBDA,weights_list)
  else:
    y = normalization(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
    svm_loss=multiclassleeloss(y,y_,10) + regularization_loss(LAMBDA,weights_list)

  global_step = tf.Variable(0, trainable=False)
  # Introducing a decaying learning rate
  learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(svm_loss, global_step=global_step)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  tps1 = time.clock() 

  #TRAIN
  for k in range(ITERATION):
    batch = mnist.train.next_batch(BATCH_SIZE)
    if k%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_:batch[1], keep_prob: 1.0})
      print("step %d, learning_rate %g, training accuracy %g"%(k, learning_rate.eval(), train_accuracy))
      if k%1000 == 0:
        print("nombre d'epochs utilisés %d"%mnist.train._epochs_completed)  
     # weights_norm = 0  
     # for weights in weights_list:
     #   weights_norm += 1*tf.cast(tf.reduce_sum(tf.square(weights)),tf.float32)
     # print("norm of weights %g" %(weights_norm))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})

  tps2 = time.clock() 
  print("weights = ")
  print(sess.run(W_fc2))
  print("biais = ")
  print(sess.run(b_fc2))
  print("sum weights = ",sess.run(tf.reduce_sum(W_fc2)))
  print("sum biais = ",sess.run(tf.reduce_sum(b_fc2)))
  print("temps training = ",int(tps2 - tps1)," secondes" )

  # TEST
  teste = mnist.test.next_batch(5000)
  print("risque réel = ",sess.run(accuracy, feed_dict={x:teste[0], y_: teste[1], keep_prob: 1.0}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
tf.app.run()
