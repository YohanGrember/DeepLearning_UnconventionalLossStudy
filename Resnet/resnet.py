import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from loss_functions import *

from config import Config

import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004 * 0.1
CONV_WEIGHT_STDDEV = 0.01 * 0.1
FC_WEIGHT_DECAY = 0.00004 * 0.1
FC_WEIGHT_STDDEV = 0.01 * 0.1
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


activation = tf.nn.relu


#def inference(x, is_training,
#              num_classes=1000,
#              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
#              use_bias=False, # defaults to using batch norm
#              bottleneck=True):
#    c = Config()
#    c['bottleneck'] = bottleneck
#    c['is_training'] = tf.convert_to_tensor(is_training,
#                                            dtype='bool',
#                                            name='is_training')
#    c['ksize'] = 3
#    c['stride'] = 1
#    c['use_bias'] = use_bias
#    c['fc_units_out'] = num_classes
#    c['num_blocks'] = num_blocks
#    c['stack_stride'] = 2
#
#    with tf.variable_scope('scale1'):
#        c['conv_filters_out'] = 64
#        c['ksize'] = 7
#        c['stride'] = 2
#        x = conv(x, c)
#        x = bn(x, c)
#        x = activation(x)
#
#    with tf.variable_scope('scale2'):
#        x = _max_pool(x, ksize=3, stride=2)
#        c['num_blocks'] = num_blocks[0]
#        c['stack_stride'] = 1
#        c['block_filters_internal'] = 64
#        x = stack(x, c)
#
#    with tf.variable_scope('scale3'):
#        c['num_blocks'] = num_blocks[1]
#        c['block_filters_internal'] = 128
#        assert c['stack_stride'] == 2
#        x = stack(x, c)
#
#    with tf.variable_scope('scale4'):
#        c['num_blocks'] = num_blocks[2]
#        c['block_filters_internal'] = 256
#        x = stack(x, c)
#
#    with tf.variable_scope('scale5'):
#        c['num_blocks'] = num_blocks[3]
#        c['block_filters_internal'] = 512
#        x = stack(x, c)
#
#    # post-net
#    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
#
#    if num_classes != None:
#        with tf.variable_scope('fc'):
#            x, last_layer_weights = fc(x, c)
#
#    return x


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x,
                    is_training,
                    num_blocks=3, # 6n+2 total weight layers will be used. num_blocks = 3 : ResNet-20. num_blocks = 5 : ResNet-32. num_blocks = 8 : ResNet-50. num_blocks = 18 : ResNet-110
                    use_bias=False, # defaults to using batch norm
                    num_classes=10):
    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['num_classes'] = num_classes
    return inference_small_config(x, c)

def inference_small_config(x, c):
    c['bottleneck'] = False
    c['ksize'] = 3
    c['stride'] = 1
    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 16
        c['block_filters_internal'] = 16
        c['stack_stride'] = 1
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)
        x = stack(x, c)

    with tf.variable_scope('scale2'):
        c['block_filters_internal'] = 32
        c['stack_stride'] = 2
        x = stack(x, c)
    with tf.variable_scope('scale3'):
        c['block_filters_internal'] = 64
        c['stack_stride'] = 2
        x = stack(x, c)

    # post-net
    x1 = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    print('x1')
    print(x1)
    
    if c['num_classes'] != None:
        with tf.variable_scope('fc'):
            x2, last_layer_weights, last_layer_biases = fc(x1, c)
            print('last_layer_weights')
            print(last_layer_weights)
            print('last_layer_biases')
            print(last_layer_biases)
            print('x2')
            print(x2)
    return x2, x1, last_layer_weights, last_layer_biases


def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(3, 3, rgb * 255.0)
    bgr = tf.concat(3, [blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr


def loss(logits, labels, basic_logits, last_layer_weights, last_layer_biases):
   
#    if FLAGS.loss == 'cross_entropy':
#        print('Loss used: Cross-Entropy')
#        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#        final_loss = tf.reduce_mean(cross_entropy)
#        
#    if FLAGS.loss == 'large_margin_softmax':
#        weight_norm=tf.sqrt(tf.reduce_sum(tf.square(weights),0))
#        x_norm=tf.sqrt(tf.reduce_sum(tf.square(basic_scores),1,keep_dims=True))
#        norm=tf.multiply(x_norm,weight_norm)
#        scalar_product=tf.matmul(basic_scores,weights)
#        cosinus=tf.divide(scalar_product,norm)
#        test2=tf.pow(tf.fill(tf.shape(scalar_product),-1.0),tf.floor(1-cosinus))
#        oo=2*tf.multiply(tf.square(cosinus),test2)-1
#        final_loss = tf.multiply(oo,norm)
#        
#    if FLAGS.loss == 'hinge':
#        print('Loss used: Hinge')
#        scores=tf.transpose(logits)
#        classes=tf.transpose(labels)
#        true_classes = tf.argmax(classes, 0)
#        idx_flattened = tf.range(0, FLAGS.batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
#        true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
#                                idx_flattened)
#        print('PRINTING NOW')
#        print(true_scores)
#        print(scores)
#        print(classes)
#        L = tf.nn.relu((1 - true_scores + scores) * (1 - tf.cast(classes,tf.float32)))
#        l=tf.reduce_sum(L,0)
#        final_loss=tf.reduce_mean(l)
   
   
   
    # Calculate the average cross entropy loss across the batch.

    labels = tf.cast(labels, tf.int64)
    print('labels')
    print(labels)
    #labels = tf.cast(labels, tf.float32)

    labels2=tf.one_hot(indices=labels,depth=FLAGS.num_classes,on_value=1.0,off_value=0.0,axis=-1)
    print('labels2')
    print(labels2)
    if FLAGS.loss=='cross_entropy':
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits= logits,  name='cross_entropy_per_example')
        loss_mean= tf.reduce_mean(cross_entropy, name='cross_entropy')
    elif FLAGS.loss=='hinge':
        loss_mean = multiclasshingeloss(logits, labels2,FLAGS.batch_size)
    elif FLAGS.loss=='crammer':
        loss_mean=multiclasscrammerloss(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'lee':
        loss_mean=multiclassleeloss(logits, labels2,10,FLAGS.batch_size)
    elif FLAGS.loss == 'surrogate_hinge':
        loss_mean = surrogate_hinge(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'surrogate_hinge_squares':
        loss_mean = surrogate_hinge_squares(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'surrogate_squares':
        loss_mean = surrogate_squares(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'surrogate_exponential':
        loss_mean = surrogate_exponential(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'surrogate_sigmoid': # bad bad results
        loss_mean = surrogate_sigmoid(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'surrogate_logistic':
        loss_mean = surrogate_logistic(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'surrogate_double_hinge': # very bad results
        loss_mean = surrogate_double_hinge(logits, labels2,FLAGS.batch_size) 
    elif FLAGS.loss == 'GEL':
        loss_mean = GEL(logits, labels2,FLAGS.batch_size)
    elif FLAGS.loss == 'GLL':
        loss_mean = GLL(logits, labels2,FLAGS.batch_size)
    elif FLAGS.loss == 'large_margin':
        loss_mean = large_margin_softmax_loss(basic_logits, labels, last_layer_weights, last_layer_biases)
    elif FLAGS.loss == 'large_margin_alex':
        scores = large_margin_scores(logits,last_layer_weights) + last_layer_biases
        loss_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits= logits,  name='cross_entropy_per_example'))




    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    tf.summary.scalar('original_loss', loss_mean)
    tf.summary.scalar('regularization_loss', tf.reduce_sum(regularization_losses))
    
    loss_ = tf.add_n([loss_mean] + regularization_losses)    
    tf.summary.scalar('total_loss', loss_)
    return loss_


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer())
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x, weights, biases


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
