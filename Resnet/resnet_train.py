from resnet import * 
from numpy import mean
import tensorflow as tf
from csv_export import *
import sys

# MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    # Counts the number of correct predictions
    num_correct = tf.reduce_sum(in_top1)
    # Returns the percentage of error on the batch
    return (batch_size - num_correct) / batch_size

# Basic_logits and last_layer_weights are added to compute Large Margin Softmax
def train(is_training, logits, images, labels, basic_logits, last_layer_weights, last_layer_biases):
    
    # Create the exports and save repositories if they don't exist
    export_folder_name='exports - loss(' + FLAGS.loss +')'
    make_sure_path_exists(export_folder_name)
    make_sure_path_exists(FLAGS.save_dir)

    # Initialize two different csv files
    test_csv_file = export_folder_name + '/' + FLAGS.export_file
    init_test_csv(test_csv_file)
    
    
    
  
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_ = loss(logits, labels, basic_logits, last_layer_weights, last_layer_biases)
    predictions = tf.nn.softmax(logits)

    top1_error = top_k_error(predictions, labels, 1)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    #opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon = 1e-8)
    grads = opt.compute_gradients(loss_)
#    print('grads')
#    print(grads)
    # We clipped the grad to avoid numerical problems with the Large_Margin Softmax Loss. Also, Nans are replaced by zeros.
    clipped_grads = [(tf.clip_by_value(tf.where(tf.is_nan(grad),tf.zeros_like(grad),grad), -1000., 1000.), var) for grad, var in grads]
#    print('clipped_grads')
#    print(clipped_grads)
    # Check to avoid updating the model with Nans
#    grad_check = tf.check_numerics(clipped_grads,'GRADIENTS EXPLODED : NOT A NUMBER','check_numerics')
#    with tf.control_dependencies([grad_check]):
#        for grad, var in clipped_grads:
#            if grad is not None and not FLAGS.minimal_summaries:
#                tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(clipped_grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()
    #summary_op = tf.constant(1)



    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.save_dir, sess.graph)

    if FLAGS.load:
        latest = tf.train.latest_checkpoint(FLAGS.load_dir)
        if not latest:
            print("No checkpoint to continue from in", FLAGS.load_dir)
            sys.exit(1)
        print("Continuing to train from", latest)
        saver.restore(sess, latest)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    
    tps1 = time.time()    
    
    for x in range(FLAGS.max_steps + 1):
        start_time = time.time()
        nb_epochs = int((x+1)*FLAGS.batch_size/50000)

        step = sess.run(global_step)
        i = [train_op, loss_]

        #write_summary = step % 100 and step > 1
        write_summary = step > 1
        if write_summary:
            i.append(summary_op)

        o = sess.run(i, { is_training: True })

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 100 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 1000 == 0:
            checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            print('Checkpoint saved in ' + FLAGS.save_dir)

        # Run validation periodically
        if step > 1 and step % 500 == 0:
            print("Processing Validation error on %d samples" %(int(9800/FLAGS.batch_size)*FLAGS.batch_size) )
            top1_error_value = mean([sess.run([val_op, top1_error], { is_training: False })[1] for _ in range(int(9800/FLAGS.batch_size))])
            print('Validation top1 error %.2f' % top1_error_value)
            csv_writerow(test_csv_file, [FLAGS.loss] + [FLAGS.batch_size] + [step] + [nb_epochs] + [1 - top1_error_value] + [time.time() - tps1])



