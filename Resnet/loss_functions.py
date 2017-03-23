# -*- coding: utf-8 -*-

import tensorflow as tf
from numpy import sqrt
#from math import pi
FLAGS = tf.app.flags.FLAGS

def normalization(scores):
    return scores-tf.tile(tf.transpose([tf.reduce_mean(scores,1)]),[1,tf.cast(scores.get_shape()[1], dtype=tf.int32)])

def regularization_loss(lamb,weights):
    return lamb*tf.reduce_sum(tf.square(weights)) 

def entropyloss(scores,classes):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=classes))

def large_margin_scores(basic_scores,weights):
  weight_norm=tf.sqrt(tf.reduce_sum(tf.square(weights),0))
  x_norm=tf.sqrt(tf.reduce_sum(tf.square(basic_scores),1,keep_dims=True))
  norm=tf.multiply(x_norm,weight_norm)

  scalar_product=tf.matmul(basic_scores,weights)
  cosinus=tf.divide(scalar_product,norm)

  test2=tf.pow(tf.fill(tf.shape(scalar_product),-1.0),tf.floor(1-cosinus))
  oo=2*tf.multiply(tf.square(cosinus),test2)-1
  return tf.multiply(oo,norm)

def large_margin_softmax_loss(x1, classes, weights, biases):
    

    def psi_by_row(cosines_row):
        return tf.map_fn(psi,cosines_row)

    def psi(cosinus, m = FLAGS.m):
        # Calculate cos(m*Theta) and then psi(Theta, m) directly from cosinus(Theta) using Chebyshev polynomials
        for k in range(0,m):
            if m == 1:
#                # x. The result should be equivalent to cross_entropy.
                return cosinus
            elif m == 2:
                # (-1)^k*cos(2*theta) - 2k
#                # 2x^2 - 1
                return tf.cond(cosinus >= 0, lambda: 2*tf.square(cosinus)-1, lambda: -(2*tf.square(cosinus)-1)-2)
            elif m == 3:
                # 4x³ - 3x
                return tf.cond(cosinus >= 0.5, lambda: 4*tf.pow(cosinus,3)-3*cosinus, 
                               lambda: tf.cond(cosinus >= -0.5, lambda: -(4*tf.pow(cosinus,3)-3*cosinus) -2, lambda:4*tf.pow(cosinus,3)-3*cosinus -4))
            elif m == 4:
                # 8x^4 -8x^2 + 1
                return tf.cond(cosinus >= sqrt(2)/2, lambda: 8*tf.pow(cosinus,4)-8*tf.pow(cosinus,2)+1, 
                               lambda: tf.cond(cosinus >= 0, lambda: -(8*tf.pow(cosinus,4)-8*tf.pow(cosinus,2)+1) - 2, 
                               lambda: tf.cond(cosinus >= -sqrt(2)/2, lambda: 8*tf.pow(cosinus,4)-8*tf.pow(cosinus,2)+1-4,lambda: -(8*tf.pow(cosinus,4)-8*tf.pow(cosinus,2)+1) - 6)))
            elif m == 5:
                # 16 x^5 - 20x^3 + 5x
                return tf.cond(cosinus >= sqrt((3+sqrt(5))/8), lambda: 16*tf.pow(cosinus,5)-20*tf.pow(cosinus,3) + 5*cosinus, 
                               lambda: tf.cond(cosinus >= sqrt((3-sqrt(5))/8), lambda : -(16*tf.pow(cosinus,5)-20*tf.pow(cosinus,3) + 5*cosinus) - 2, 
                               lambda: tf.cond(cosinus >= -sqrt((3-sqrt(5))/8), lambda : (16*tf.pow(cosinus,5)-20*tf.pow(cosinus,3) + 5*cosinus) - 4,
                               lambda: tf.cond(cosinus >= -sqrt((3+sqrt(5))/8), lambda: -(16*tf.pow(cosinus,5)-20*tf.pow(cosinus,3) + 5*cosinus) - 6,lambda: (16*tf.pow(cosinus,5)-20*tf.pow(cosinus,3) + 5*cosinus) - 8))))   
            elif m == 6:
                # 32 x^6 - 48 x^4 + 18x^2 - 1
                return tf.cond(cosinus >= sqrt(3)/2, lambda: 32*tf.pow(cosinus,6)-48*tf.pow(cosinus,4) + 18*tf.pow(cosinus,2)-1, 
                               lambda: tf.cond(cosinus >= 0.5, lambda: -(32*tf.pow(cosinus,6)-48*tf.pow(cosinus,4) + 18*tf.pow(cosinus,2)-1) - 2, 
                               lambda: tf.cond(cosinus >= 0, lambda : (32*tf.pow(cosinus,6)-48*tf.pow(cosinus,4) + 18*tf.pow(cosinus,2)-1) - 4, 
                               lambda: tf.cond(cosinus >= -0.5, lambda : -(32*tf.pow(cosinus,6)-48*tf.pow(cosinus,4) + 18*tf.pow(cosinus,2)-1) - 6,
                               lambda: tf.cond(cosinus >= -sqrt(3)/2, lambda: (32*tf.pow(cosinus,6)-48*tf.pow(cosinus,4) + 18*tf.pow(cosinus,2)-1) - 8,lambda: -(32*tf.pow(cosinus,6)-48*tf.pow(cosinus,4) + 18*tf.pow(cosinus,2)-1) - 10)))))
            else :
                print('please choose integer m between 1 and 6')
       
    print('Loss used: Large Margin Softmax with m = %d' %FLAGS.m)
    
    x1_norms = tf.norm(x1,axis=1,keep_dims=True)
    classwise_weights_norms = tf.norm(weights,axis=0,keep_dims=True)

    x2_max_scores=tf.multiply(x1_norms,classwise_weights_norms)
    x2 = tf.matmul(x1, weights)

    cosines=tf.divide(x2,x2_max_scores)
    psi_matrix = tf.map_fn(psi_by_row,cosines, name = 'psi_calculation')

    labels=tf.one_hot(indices=classes,depth=10,on_value=True,off_value=False,axis=-1)

    large_margin_cosines_matrix = tf.where(labels, psi_matrix, cosines, name='margin_enlargement')
#    large_margin_scores_unbiased = tf.multiply(large_margin_cosines_matrix,x2_max_scores)
    large_margin_scores = tf.multiply(large_margin_cosines_matrix,x2_max_scores) + biases
    safe_scores = tf.subtract(large_margin_scores,tf.reduce_max(tf.expand_dims(large_margin_scores,-1),axis=1))

    #logsoftmax_losses = -tf.nn.log_softmax(safe_scores, dim=0)
    logsoftmax_2D_losses = tf.subtract(tf.log(tf.expand_dims(tf.reduce_sum(tf.exp(safe_scores), axis = 1),-1)), safe_scores)
    logsoftmax_losses = tf.reduce_sum(tf.where(labels, logsoftmax_2D_losses, tf.zeros_like(logsoftmax_2D_losses),name='logsoftmax_selection'),axis=1)
    loss_mean = tf.reduce_mean(logsoftmax_losses)
    
    return loss_mean
    
    

def multiclasshingeloss(scores, classes,batch_size):
    print('Loss used: Multiclass Hinge Loss')
    scores=tf.transpose(normalization(scores))
    classes=tf.transpose(classes)
    true_classes = tf.argmax(classes, 0)
    idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
    true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                            idx_flattened)
    L = tf.nn.relu((1 - true_scores + scores) * (1 - classes))
    return tf.reduce_mean(tf.reduce_sum(L,0))

def multiclasscrammerloss(scores, classes,batch_size):
    print('Loss used: Multiclass Crammer Loss')
    scores=tf.transpose(normalization(scores))
    classes=tf.transpose(classes)
    true_classes = tf.argmax(classes, 0)
    idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
    true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                            idx_flattened)
    L=tf.nn.relu(1-true_scores + tf.reduce_max(scores*(1-classes),reduction_indices=[0]))
    final_loss=tf.reduce_mean(L)
    return final_loss

def multiclassleeloss(scores,classes,nb_class,batch_size):
    print('Loss used: Multiclass Lee Loss')
    scores=tf.transpose(normalization(scores))
    classes=tf.transpose(classes)
    L = tf.nn.relu( (1/(nb_class-1) + scores) * (1 - classes))
    l = tf.reduce_sum(L,0)
    final_loss=tf.reduce_mean(l)
    return final_loss

def surrogate_hinge(scores, classes,batch_size):
    print('Loss used: Surrogate Hinge Loss')
    scores = tf.transpose(normalization(scores))
    classes = tf.transpose(classes)
    L = tf.nn.relu((1 + scores)*(1-classes))
    l = tf.reduce_sum(L, 0)
    final_loss = tf.reduce_mean(l)
    return final_loss

def surrogate_hinge_squares(scores, classes,batch_size):
    print('Loss used: Surrogate Hinge Squares Loss')
    scores = tf.transpose(normalization(scores))
    classes = tf.transpose(classes)
    L = tf.nn.relu((1 + scores)*(1-classes))
    l = tf.reduce_sum(tf.square(L), 0)
    final_loss = tf.reduce_mean(l)
    return final_loss

def surrogate_squares(scores, classes,batch_size):
    print('Loss used: Surrogate Squares Loss')
    scores = tf.transpose(normalization(scores))
    classes = tf.transpose(classes)
    L = (1 + scores)*(1-classes)
    l = tf.reduce_sum(tf.square(L), 0)
    final_loss = tf.reduce_mean(l)
    return final_loss

def surrogate_exponential(scores, classes,batch_size):
    print('Loss used: Surrogate Exponential Loss')
    scores = tf.transpose(normalization(scores))
    classes = tf.transpose(classes)
    L = tf.exp(scores)*(1-classes)
    l = tf.reduce_sum(L, 0)
    final_loss = tf.reduce_mean(l)
    return final_loss

def surrogate_sigmoid(scores, classes,batch_size):
    print('Loss used: Surrogate Sigmoid Loss')
    scores = tf.transpose(normalization(scores))
    classes = tf.transpose(classes)
    L = (1+tf.tanh(scores))*(1-classes)
    l = tf.reduce_sum(L, 0)
    final_loss = tf.reduce_mean(l)
    return final_loss

def surrogate_logistic(scores, classes,batch_size):
    print('Loss used: Surrogate Logistic Loss')
    scores = tf.transpose(normalization(scores))
    classes = tf.transpose(classes)
    L = tf.log(1+tf.exp(scores))*(1-classes)
    l = tf.reduce_sum(L, 0)
    final_loss = tf.reduce_mean(l)
    return final_loss

def surrogate_double_hinge(scores, classes,batch_size):
    print('Loss used: Surrogate Double Hinge Loss')
    scores = tf.transpose(normalization(scores))
    classes = tf.transpose(classes)
    L = tf.subtract(tf.nn.relu(1 + scores),tf.nn.relu(scores))*(1-classes)
    l = tf.reduce_sum(L, 0)
    final_loss = tf.reduce_mean(l)
    return final_loss

def GEL(scores, classes, batch_size):
    print('Loss used: Generalized Exponential Loss')
    scores=tf.transpose(normalization(scores))
    #scores=tf.cast(scores, tf.float64)
    classes=tf.transpose(classes)
    true_classes = tf.argmax(classes, 0)
    idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
    true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                            idx_flattened)
    L = tf.exp(scores - true_scores)*(1 - classes)
    l = tf.reduce_sum(L, 0)
    return tf.reduce_mean(l)

#FAUSSE ?
def GLL(scores, classes, batch_size):
    print('Loss used: Generalized Exponential Loss')
    scores=tf.transpose(normalization(scores))
    classes=tf.transpose(classes)
    true_classes = tf.argmax(classes, 0)
    idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
    true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                            idx_flattened)
    L = tf.exp(scores - true_scores)*(1 - classes)
    l = tf.log1p(tf.reduce_sum(L, 0))
    final_loss = tf.reduce_mean(l)
    return final_loss
