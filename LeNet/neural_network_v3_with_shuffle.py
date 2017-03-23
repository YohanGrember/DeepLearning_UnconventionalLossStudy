# This script train and test a deep CNN on MNIST, cifar and icml database

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import time
import math

from loss_functions import *
from csv_export import *
from handle_datasets import *

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
#from sklearn.utils import shuffle
import numpy as np

def shuffle(X_train, Y_train):
    # Creates a random order for images
    a = np.arange(len(X_train))
    np.random.shuffle(a)
    # Set the images and labels in the same, new order
    return X_train.take(a,0), Y_train.take(a,0)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    if DATASET == 'mnist':
        mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
        X_train, Y_train = mnist.train.images, mnist.train.labels
        X_test, Y_test = mnist.test.next_batch(5000)

        picture_height = 28
        nb_train = len(Y_train)
        input_len = 784
        nb_classes = 10
        nb_colors = 1

    elif DATASET == 'icml':
        X, Y = get_icml('fer2013.csv')
        nb_train = 28709
        X_train, X_test = X[:nb_train, :], X[(nb_train + 1):(nb_train + 800), :]
        Y_train, Y_test = Y[:nb_train, :], Y[(nb_train + 1):(nb_train + 800), :]

        nb_classes = 7
        input_len = 2304
        picture_height = 48
        nb_colors = 1

    elif DATASET == 'cifar10':
        X_train, Y_train = get_cifar_train()
        X_test_all, Y_test_all = get_cifar_test()
        X_test, Y_test = X_test_all[:500, :], Y_test_all[:500, :]

        input_len = 32
        picture_height = 32
        nb_train = len(Y_train)
        nb_classes = 10
        nb_colors = 3

    if DATASET == 'cifar10':
        x = tf.placeholder(tf.float32, [None, picture_height, picture_height, 3])
        x_image = x
    else:
        x = tf.placeholder(tf.float32, [None, picture_height * picture_height])
        x_image = tf.reshape(x, [-1, picture_height, picture_height, 1])

    # first layer
    W_conv1 = weight_variable([5, 5, nb_colors, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([int(picture_height * picture_height / 16) * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, int(picture_height * picture_height / 16) * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # drop out
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # #readout layer
    W_fc2 = weight_variable([1024, nb_classes])
    b_fc2 = bias_variable([nb_classes])

    y_ = tf.placeholder(tf.float32, [None, nb_classes])

    weights_list = W_fc2  # [W_conv1, W_conv2, W_fc1, W_fc2]

    # define the loss and gradient descent
    if LOSS == 'entropy':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = entropyloss(y, y_) + regularization_loss(LAMBDA, weights_list)
    #   svm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = y_) + regularization_loss(LAMBDA, weights_list)
    if LOSS == 'hinge':
        y = normalization(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        print('Passage ici')
        svm_loss = multiclasshingeloss(y, y_, BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'crammer':
        y = normalization(tf.matmul(h_fc1_drop, W_fc2))
        svm_loss = multiclasscrammerloss(y, y_, BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'lee':
        y = normalization(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        svm_loss = multiclassleeloss(y, y_, nb_classes, BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    # elif LOSS == 'surrogate':
    #     y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    #     svm_loss = surrogateloss(y, y_, BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'surrogate_hinge':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = surrogate_hinge(y, y_,BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'surrogate_hinge_squares':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = surrogate_hinge_squares(y, y_,BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'surrogate_squares':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = surrogate_squares(y, y_,BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'surrogate_exponential':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = surrogate_exponential(y, y_,BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'surrogate_sigmoid': # bad bad results
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = surrogate_sigmoid(y, y_,BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'surrogate_logistic':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = surrogate_logistic(y, y_,BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'surrogate_double_hinge': # very bad results
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = surrogate_double_hinge(y, y_,BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'GEL':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = GEL(y, y_, BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'GLL':
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = GLL(y, y_, BATCH_SIZE) + regularization_loss(LAMBDA, weights_list)
    elif LOSS == 'large_margin_entropy':
        y = large_margin_scores(h_fc1_drop, W_fc2) + b_fc2
        svm_loss = entropyloss(y, y_) + regularization_loss(LAMBDA, weights_list)

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(svm_loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Create the exports and sessions repositories if they don't exist
    export_folder_name = 'exports - loss(' + LOSS + ') le ' + time.strftime("%d-%m-%Y à %H:%M")
    session_folder_name = 'sessions - loss(' + LOSS + ') le ' + time.strftime("%d-%m-%Y à %H:%M")
    make_sure_path_exists(export_folder_name)
    make_sure_path_exists(session_folder_name)

    # Initialize two different csv files
    test_csv_file = export_folder_name + '/test.csv'
    init_test_csv(test_csv_file)

    tps1 = time.time()

    # TRAIN
    for k in range(ITERATION + 1):

        batch = np.arange(k * BATCH_SIZE, (k + 1) * BATCH_SIZE) % nb_train  # batchs successifs
        batch_xs, batch_ys = X_train[batch, :], Y_train[batch, :]

        # shuflle when the dataset has been browse
        if k * (BATCH_SIZE + 1) % nb_train > (k + 1) * (BATCH_SIZE + 1) % nb_train:
            X_train, Y_train = shuffle(X_train, Y_train)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        if k % 1000 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
            nb_epochs = int((k + 1) * BATCH_SIZE / nb_train)

            print("nombre d'epochs utilisés %d" % nb_epochs)
            print("step %d, training accuracy %g" % (k, test_accuracy))
            # write in the csv file the current accuracy
            csv_writerow(test_csv_file, [LOSS] + [k] + [nb_epochs] + [test_accuracy] + [time.time() - tps1])
            # save the weights
            saver.save(sess, session_folder_name + '/Session-MNIST-Iteration-%d-epoch-%d' % (k, nb_epochs))

    tps2 = time.time()
    print("temps training = ", int(tps2 - tps1), " secondes")

    # TEST
    test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
    print("test_accuracy = %g" % test_accuracy)

    nb_epochs = int((k + 1) * BATCH_SIZE / nb_train)
    print("nombre d'epochs utilisés %d" % nb_epochs)

    csv_writerow(test_csv_file, [LOSS] + [k] + [nb_epochs] + [test_accuracy] + [time.time() - tps1])
    saver.save(sess, session_folder_name + '/Session-MNIST-Iteration-%d-epoch-%d' % (k, nb_epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/data',
    #                     help='Directory for storing data')
    parser.add_argument('-data', '--dataset', type=str, default='mnist', help='dataset : mnist / cifar10 / icml')
    parser.add_argument('-loss', '--loss', type=str, default='hinge',
                        help='loss function : entropy / hinge / crammer / lee / surrogate / GEL / GLL / large-margin entropy')
    parser.add_argument('-batchsize', '--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('-nb_iter', '--nb_iterations', type=int, default=30, help='nombre diterations')
    parser.add_argument('-lambda', '--lamb', type=float, default=0.5, help='lambda coeff for regularisation')
    parser.add_argument('-learn_rate', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for gradient descent')
    # FLAGS = parser.parse_args()
    args = parser.parse_args()

    DATASET = args.dataset
    LOSS = args.loss
    BATCH_SIZE = args.batch_size
    ITERATION = args.nb_iterations
    LAMBDA = args.lamb  # devant le terme de régularization
    LEARNING_RATE = args.learning_rate

    tf.app.run()
