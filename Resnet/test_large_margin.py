import tensorflow as tf
from numpy import sqrt
from numpy import cos, arccos
import numpy as np
from math import pi



def psi(cosinus, m = 2):
    if True:
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

def psi_by_row(cosines_row):
    return tf.map_fn(psi,cosines_row)

matrix = np.random.randn(4,6)
print(matrix)

classes = tf.constant([1,5,9,0])
x1 = tf.Variable(tf.random_normal([4,6], stddev=0.35),
                      name="x1")
weights = tf.Variable(tf.random_normal([6,10], stddev=0.1),
                      name="weights")
biases = tf.Variable(tf.random_normal([10], stddev=0.1),
                      name="biases")

x1_norms = tf.norm(x1,axis=1,keep_dims=True)
classwise_weights_norms = tf.norm(weights,axis=0,keep_dims=True)

x2_max_scores=tf.multiply(x1_norms,classwise_weights_norms)
x2 = tf.matmul(x1, weights)

cosines=tf.divide(x2,x2_max_scores)
psi_matrix = tf.map_fn(psi_by_row,cosines, name = 'psi_calculation')

#labels = tf.one_hot(labels, 10, on_value=True, off_value=False, axis=-1) 
labels=tf.one_hot(indices=classes,depth=10,on_value=True,off_value=False,axis=-1)

large_margin_cosines_matrix = tf.where(labels, psi_matrix, cosines, name='margin_enlargement')
large_margin_scores_unbiased = tf.multiply(large_margin_cosines_matrix,x2_max_scores)
large_margin_scores = tf.multiply(large_margin_cosines_matrix,x2_max_scores) + biases
safe_scores = tf.subtract(large_margin_scores,tf.reduce_max(tf.expand_dims(large_margin_scores,-1),axis=1))

#logsoftmax_losses = -tf.nn.log_softmax(safe_scores, dim=0)
logsoftmax_losses = tf.subtract(tf.log(tf.expand_dims(tf.reduce_sum(tf.exp(safe_scores), axis = 1),-1)), safe_scores)
logsoftmax_losses_2 = tf.reduce_sum(tf.where(labels, logsoftmax_losses, tf.zeros_like(logsoftmax_losses),name='logsoftmax_selection'),axis = 1)
loss_mean = tf.reduce_mean(logsoftmax_losses_2)

theta = pi/4
cosinus = tf.constant(cos(theta))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))


    
with tf.Session() as sess:
#    print ('theta = %3f' %theta)
#    print(cos(theta))
#    print(sess.run(cosinus))
#    print(sess.run(psi(cosinus,2)))
#    print(cos(2*theta))
#    print(sess.run(psi(cosinus,3)))
#    print(cos(3*theta))
#    print(sess.run(psi(cosinus,4)))
#    print(cos(4*theta))
#    print(sess.run(psi(cosinus,5)))
#    print(cos(5*theta))
#    print(sess.run(psi(cosinus,6)))
#    print(cos(6*theta))
#    print(cos(6*theta))
    init = tf.global_variables_initializer()
    sess.run(init)
    print('x1')
    print(sess.run(x1))
    print('x1_norms')
    print(sess.run(x1_norms))
    print('weights')
    print(sess.run(weights))
    print('classwise_weights_norms')
    print(sess.run(classwise_weights_norms))
    print('x2_max_scores')
    print(sess.run(x2_max_scores))
    print('x2')
    print(sess.run(x2))
    print('cosines')
    print(sess.run(cosines))
    print('psi_matrix')
    print(sess.run(psi_matrix))
    print('classes')
#    print(classes)
    print(sess.run(classes))
    print('labels')
#    print(labels)
    print(sess.run(labels))
    print('large_margin_cosines_matrix-cosinus')
    print(sess.run(large_margin_cosines_matrix-cosines))
    print('where_matrix - psi_matrix')
    print(sess.run(large_margin_cosines_matrix-psi_matrix))
    print('large_margin_scores_unbiased')
#    print(large_margin_scores_unbiased)
    print(sess.run(large_margin_scores_unbiased))
    print('large_margin_scores')
#    print(large_margin_scores)
    print(sess.run(large_margin_scores))
    print('safe_scores')
#    print(safe_scores)
    print(sess.run(safe_scores))
    print('logsoftmax_losses')
#    print(logsoftmax_losses)
    print(sess.run(logsoftmax_losses))
    print('logsoftmax_losses_2')
#    print(logsoftmax_losses)
    print(sess.run(logsoftmax_losses_2))
    print('loss_mean')
#    print(loss_mean)
    print(sess.run(loss_mean))
    
    
#        print('Loss used: Large Margin Softmax with m = %d' %FLAGS.m)
#    print(x1)
#    print(weights)
#    classwise_weights_norms=tf.sqrt(tf.reduce_sum(tf.square(weights),0,keep_dims=True))
#    tf.summary.tensor_summary('classwise_weights_norms', classwise_weights_norms)
#    
#    print('classwise_weights_norms')
#    print(classwise_weights_norms)
#    x1_norms=tf.sqrt(tf.reduce_sum(tf.square(x1),1,keep_dims=True))
#    tf.summary.tensor_summary('x1_norms', x1_norms)
#    print('x1_norms')
#    print(x1_norms)
#    x2_max_scores=tf.multiply(x1_norms,classwise_weights_norms)
#    tf.summary.tensor_summary('x2_max_scores', x2_max_scores)
#    print('x2_max_scores')
#    print(x2_max_scores)
#    x2 = tf.matmul(x1, weights)
#    cosines=tf.divide(x2,x2_max_scores)
#    tf.summary.tensor_summary('cosines', cosines)
#    print('cosines')
#    print(cosines)
#    
#    psi_matrix = tf.map_fn(psi_by_row,cosines, name = 'psi_calculation')
#        
#    labels = tf.one_hot(classes, FLAGS.num_classes, on_value=True, off_value=False, axis=-1)  
#    print('labels')
#    print(labels)
#        
#    large_margin_scores = tf.multiply(tf.where(labels, psi_matrix, cosines, name='margin_enlargement'),x2_max_scores) + biases
#    safe_scores = tf.subtract(large_margin_scores,tf.reduce_max(tf.expand_dims(large_margin_scores,-1), axis = 1))
##    logsoftmax_losses = tf.subtract(tf.log(tf.expand_dims(tf.reduce_sum(tf.exp(safe_scores), axis = 1),-1)), safe_scores)
#    logsoftmax_losses = -tf.nn.log_softmax(safe_scores)
#    return tf.reduce_mean(logsoftmax_losses)
    
    
