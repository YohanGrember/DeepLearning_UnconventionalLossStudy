# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division

import os
import sys
import tarfile
import argparse

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

from resnet_train import train
from resnet import inference_small
import tensorflow as tf
import numpy as np

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.slice(record_bytes, [label_bytes], [image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    filenames = [
        os.path.join(data_dir, 'cifar-10-batches-bin', 'data_batch_%d.bin' % i)
        for i in xrange(1, 6)
    ]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(
        distorted_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    if not eval_data:
        assert False, "hack. shouldn't go here"
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'cifar-10-batches-bin', 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=False)


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(argv=None):  # pylint: disable=unused-argument
    maybe_download_and_extract()

    images_train, labels_train = distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
    images_val, labels_val = inputs(True, FLAGS.data_dir, FLAGS.batch_size)
    


    is_training = tf.placeholder('bool', [], name='is_training')

    images, labels = tf.cond(is_training,
        lambda: (images_train, labels_train),
        lambda: (images_val, labels_val))
    

    logits, basic_logits, last_layer_weights, last_layer_biases = inference_small(images,
                             num_classes=FLAGS.num_classes,
                             is_training=is_training,
                             use_bias=(not FLAGS.use_bn),
                             num_blocks=FLAGS.num_blocks)
    train(is_training, logits, images, labels, basic_logits, last_layer_weights, last_layer_biases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir','--data_dir', type=str, default='./data',
                         help='Directory for storing the datasets')
    parser.add_argument('-save_dir','--save_dir', type=str, default='./save',
                         help='Directory for writing event logs and checkpoints')
    parser.add_argument('-export_file','--export_file', type=str, default='test.csv',
                         help='name of the csv export file')
    #parser.add_argument('-data','--dataset',type=str,default='mnist',help='dataset : mnist / cifar10 / icml')
    parser.add_argument('-loss','--loss',type=str,default='cross_entropy',help='loss function : cross_entropy / weston / crammer / lee / surrogate_hinge / surrogate_hinge_squares / surrogate_squares / surrogate_exponential / surrogate_sigmoid / surrogate_logistic / surrogate_saturated_hinge / GEL / GLL / large_margin')
    parser.add_argument('-batch_size','--batch_size',type=int,default=128,help='batch size')
    parser.add_argument('-max_steps','--max_steps',type=int,default=200000,help='nombre diterations')
    parser.add_argument('-num_blocks','--num_blocks',type=int,default=3,help='6n+2 total weight layers will be used. num_blocks = 3 : ResNet-20. num_blocks = 5 : ResNet-32. num_blocks = 8 : ResNet-50. num_blocks = 18 : ResNet-110')
    #parser.add_argument('-lambda','--lamb',type=float,default=0.5,help='lambda coeff for regularisation')
    parser.add_argument('-learn_rate','--learning_rate',type=float,default=3e-4,help='learning rate for gradient descent')
    parser.add_argument('-m','--m',type=int,default=2,help='margin coefficient for Large Margin Softmax Loss')
    parser.add_argument('-load','--load',type=bool,default=False, help='Initialize the network from a checkpoint ?')
    parser.add_argument('-load_dir','--load_dir',type=str,default='./checkpoint_to_load',help='Directory from which to load the network')
    
    args=parser.parse_args()

    tf.app.flags.DEFINE_string('data_dir', args.data_dir,
                               """Directory where to store the datasets.""")
    tf.app.flags.DEFINE_string('save_dir', args.save_dir + ' - loss(' + args.loss + ')',
                               """Directory where to write event logs """
                               """and checkpoint.""")
    tf.app.flags.DEFINE_string('export_file', args.export_file,
                               """Name of the csv export file""")
    tf.app.flags.DEFINE_string('loss', args.loss, "loss")
    tf.app.flags.DEFINE_float('learning_rate', args.learning_rate, "learning rate.")
    tf.app.flags.DEFINE_integer('batch_size', args.batch_size, "batch size")
    tf.app.flags.DEFINE_integer('max_steps', args.max_steps, "max steps")
    tf.app.flags.DEFINE_integer('num_blocks', args.num_blocks, "num_blocks")
    tf.app.flags.DEFINE_integer('num_classes', 10, "num_classes")
    tf.app.flags.DEFINE_integer('m', args.m, "m")
    tf.app.flags.DEFINE_boolean('load', args.load, "load")
    tf.app.flags.DEFINE_string('load_dir', args.load_dir + ' - loss(' + args.loss + ')', "load_dir")
    
tf.app.run()
