import tensorflow as tf

def normalization(scores):
  return scores-tf.tile(tf.transpose([tf.reduce_mean(scores,1)]),[1,tf.cast(scores.get_shape()[1], dtype=tf.int32)])

def regularization_loss(lamb,weights):
  return lamb*tf.reduce_sum(tf.square(weights)) 

def entropyloss(scores,classes):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, classes))

def large_margin_scores(basic_scores,weights):
  weight_norm=tf.sqrt(tf.reduce_sum(tf.square(weights),0))
  x_norm=tf.sqrt(tf.reduce_sum(tf.square(basic_scores),1,keep_dims=True))
  norm=tf.multiply(x_norm,weight_norm)

  scalar_product=tf.matmul(basic_scores,weights)
  cosinus=tf.divide(scalar_product,norm)

  test2=tf.pow(tf.fill(tf.shape(scalar_product),-1.0),tf.floor(1-cosinus))
  oo=2*tf.multiply(tf.square(cosinus),test2)-1
  return tf.multiply(oo,norm)

def multiclasshingeloss(scores, classes,batch_size):
  scores=tf.transpose(scores)
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.nn.relu((1 - true_scores + scores) * (1 - classes))
  l=tf.reduce_sum(L,0)
  final_loss=tf.reduce_mean(l)
  return final_loss

def multiclasscrammerloss(scores, classes,batch_size):
  scores=tf.transpose(scores)
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L=tf.nn.relu(1-true_scores + tf.reduce_max(scores*(1-classes),reduction_indices=[0]))
  final_loss=tf.reduce_mean(L)
  return final_loss

def multiclassleeloss(scores,classes,nb_class,batch_size):
  scores=tf.transpose(scores)
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.nn.relu( (1/(nb_class-1) + scores) * (1 - classes))
  l=tf.reduce_sum(L,0)
  final_loss=tf.reduce_mean(l)
  return final_loss


def surrogateloss(scores, classes,batch_size):
  scores = tf.transpose(scores)
  classes = tf.transpose(classes)
  cost = 1;
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0] + tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = cost * tf.nn.relu(1 - true_scores + scores)
  l = tf.reduce_sum(L, 0)
  final_loss = tf.reduce_mean(l)
  return final_loss

def GLE(scores, classes, nb_class,batch_size):
  scores = tf.transpose(scores)
  classes = tf.transpose(classes)
  cost = [[1.0 for x in range(nb_class)] for x in range(nb_class)]
  for i in range(nb_class) :
    for j in range(nb_class):
      if i==j:
          cost[i][j]=0.0

  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0] + tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.matmul(cost, tf.exp(scores - true_scores))
  #L = cost * tf.exp(scores - true_scores)
  #L = tf.exp(scores - true_scores)
  l = tf.reduce_sum(L, 0)
  final_loss = tf.reduce_mean(l)
  return final_loss

def GLL(scores, classes, nb_class,batch_size):
  scores = tf.transpose(scores)
  classes = tf.transpose(classes)
  cost = [[1.0 for x in range(nb_class)] for x in range(nb_class)]
  for i in range(nb_class) :
    for j in range(nb_class):
      if i==j:
          cost[i][j]=0.0

  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0] + tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.matmul(cost, tf.exp(scores - true_scores))
  l = tf.reduce_sum(L, 0)
  final_loss = tf.reduce_mean(l)
  final_loss2 = tf.log(1+final_loss)
  return final_loss2