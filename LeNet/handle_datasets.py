import pandas as pd
import numpy as np
import pickle

######################## for ICML ###################################
def get_icml(filename):
  df=pd.read_csv('fer2013.csv')
  a=df.as_matrix(['pixels'])
  c=[]

  for i in range(0,len(a)):
    b=np.fromstring(a[i,0],dtype=int,sep=' ')
    c.append(b)
  d=np.asarray(c)

  p=pd.get_dummies(df['emotion'])
  l=p.as_matrix(columns=[p.columns[:]])
  return d,l

######################## for CIFAR10 ###################################
def get_dummies(y, n_labels):
  y_train = np.zeros((len(y),n_labels), dtype=int)
  for i in range(len(y)):
    y_train[i][y[i]] = 1
  return y_train

def _unpickle(filename):
  with open(filename, mode='rb') as file:
    data = pickle.load(file, encoding='bytes')
  return data

def _convert_images(raw):
  raw_float = np.array(raw, dtype=float) / 255.0
  images = raw_float.reshape([-1, 3, 32, 32])
  images = images.transpose([0, 2, 3, 1])
  return images

def _load_data(filename):
  data = _unpickle(filename)
  raw_images = data[b'data']
  labels = np.array(data[b'labels'])
  images = _convert_images(raw_images)
  return images, labels

def get_cifar_train():
  images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
  labels = np.zeros(shape=[50000], dtype=int)
  begin = 0

  for i in range(5):
    images_batch, cls_batch = _load_data(filename='cifar-10-batches-py/data_batch_' + str(i + 1))
    num_images = len(images_batch)
    end = begin + num_images
    images[begin:end, :] = images_batch
    labels[begin:end] = cls_batch
    begin = end

  return images,  get_dummies(labels, 10)

def get_cifar_test():
  images, labels = _load_data(filename='cifar-10-batches-py/test_batch')
  return images, get_dummies(labels, 10)