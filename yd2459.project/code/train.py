import os
import json
from datetime import datetime
import time

import tensorflow as tf
from TFrecords_Preparation import encoder_main, Batch_Generation
from model import Model


#--------------------------------------------------
#             General Param Setting
#--------------------------------------------------
tf.app.flags.DEFINE_integer('epoch', 20, 'Default 20')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Default 32')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Default 1e-2')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Default 10000')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Default 0.9')

tf.app.flags.DEFINE_string('data_dir', './data', 'Dir: read TFRecords files')
tf.app.flags.DEFINE_string('train_log_dir', './logs/train', 'Dir: write training logs')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint e.g. ./logs/train/model.ckpt-100')
tf.app.flags.DEFINE_string('checkpoint_dir', './logs/train', 'Dir: read checkpoint files')
tf.app.flags.DEFINE_string('eval_log_dir', './logs/eval', 'Dir: write evaluation logs')
FLAGS = tf.app.flags.FLAGS

#--------------------------------------------------
#                  Train Model       
#--------------------------------------------------
def train(train_tfrecords_file, num_train_examples, val_tfrecords_file, num_val_examples,
           LogTrain_dir, restored_checkpoint, training_param, model_param):
    epoch = training_param['epoch']
    batch_size = training_param['batch_size']
    num_steps_show_info = 200

    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        #get batch
        img_batch, length_batch, digits_batch = Batch_Generation(train_tfrecords_file,num_examples=num_train_examples,
                                                                   batch_size=batch_size,shuffled=True)
        length, digits = Model.inference(img_batch, model_param, drop_rate=0.2)

        #get loss
        loss = Model.loss(length, digits, length_batch, digits_batch)

        #set optimizer
        learning_rate = tf.train.exponential_decay(training_param['learning_rate'], global_step=global_step,
                                                   decay_steps=training_param['decay_steps'],
                                                   decay_rate=training_param['decay_rate'], staircase=True)
        
        #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        tf.summary.image('image', img_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer_train = tf.summary.FileWriter(LogTrain_dir, sess.graph)
            path_to_eval_log_dir = os.path.join(LogTrain_dir, 'eval/val')
            summary_writer_val = tf.summary.FileWriter(path_to_eval_log_dir)

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            #set restore options
            saver = tf.train.Saver()
            if restored_checkpoint is not None:
                assert tf.train.checkpoint_exists(restored_checkpoint), \
                    '%s not found' % restored_checkpoint
                saver.restore(sess, restored_checkpoint)
                print ('Model restored from file: %s' % restored_checkpoint)

            #start training
            print ('Start training')
            best_accuracy = 0.0
            iter_per_ep=int(num_train_examples/batch_size)

            for epc in range(epoch):
                print('epoch:',epc)

                #in each epoch, do optimization
                for iter in range(iter_per_ep):
                    _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([optimizer, loss, summary, global_step, learning_rate])
                    #lr=tf.to_float(learning_rate_val)

                    #show loss every 200 steps
                    if global_step_val % num_steps_show_info == 0:
                        print ('epoch: %d: %s: step %d, loss = %f,' % (
                            epc,datetime.now(), global_step_val, loss_val ))
                        print('learning_rate:',learning_rate_val)
                        print('current best acc:',best_accuracy)

                    summary_writer_train.add_summary(summary_val, global_step=global_step_val)

                #after each epoch, do validation
                print ('Validation.')
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(LogTrain_dir, 'latest.ckpt'))
                accuracy = evaluate(summary_writer_val,path_to_latest_checkpoint_file,val_tfrecords_file,
                                    num_val_examples,global_step_val,model_param)
                print ('Accuracy: %f, Best accuracy: %f' % (accuracy, best_accuracy))

                #save best model
                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(LogTrain_dir, 'model.ckpt'), global_step=global_step_val)
                    print ('Model saved: %s' % path_to_checkpoint_file)
                    best_accuracy = accuracy

            coord.request_stop()
            coord.join(threads)
            print ('Training Completed')


#--------------------------------------------------
#         Validation after each epoch
#--------------------------------------------------
def evaluate(summary_writer, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step, model_param):
    batch_size = 128
    #batch_size=num_examples
    num_batches = int(num_examples / batch_size)

    with tf.Graph().as_default():
        img_batch, length_batch, digits_batch = Batch_Generation(path_to_tfrecords_file,num_examples=num_examples,
                                                                   batch_size=batch_size,shuffled=False)    
        length_logits, digits_logits = Model.inference(img_batch, model_param, drop_rate=0.0)
        length_predictions = tf.argmax(length_logits, axis=1)
        digits_predictions = tf.argmax(digits_logits, axis=2)

        labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
        predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)

        labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
        predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

        accuracy, update_accuracy = tf.metrics.accuracy(
            labels=labels_string,
            predictions=predictions_string
        )

        tf.summary.image('image', img_batch)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('variables',tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))            
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            restorer = tf.train.Saver()
            restorer.restore(sess, path_to_checkpoint)

            for bat in range(num_batches):
                sess.run(update_accuracy)

            accuracy_val, summary_val = sess.run([accuracy, summary])
            summary_writer.add_summary(summary_val, global_step=global_step)

            coord.request_stop()
            coord.join(threads)

    return accuracy_val

#--------------------------------------------------
#           Evaluation after training
#--------------------------------------------------
def eval_total(checkpoint_dir, eval_tfrecords_file, num_eval_examples, eval_log_dir, model_param):
    
    summary_writer_test = tf.summary.FileWriter(eval_log_dir)
    checkpoint_paths = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue

        accuracy = evaluate(summary_writer_test, path_checkpoint, eval_tfrecords_file, num_eval_examples, global_step_val, model_param)
        print ('Evaluate %s on %s, accuracy = %f' % (path_checkpoint, eval_tfrecords_file, accuracy))


#--------------------------------------------------
#                 Main Function
#--------------------------------------------------
def main(_):
    
    #encoder_main() #Transfer the Original PNG file into TFrecord file used as the model input

    #get tf records
    train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    Train_create = str(datetime.now())
    LogTrain_dir = FLAGS.train_log_dir+Train_create
    restored_checkpoint = FLAGS.restore_checkpoint

    #set training param & model param
    model_param = {
        'filters_num': [48,64,128,160,192,192,192,192],
        'kernel_size': [5,5,5,5,3,3,3,3],
        'pool_size': [2,2,2,2,2,2,2,2],
        'strides': [2,1,2,1,2,1,2,1],
        'dense': [3072,3072],
        'flatten': [4,4]
    }
    training_param = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate,
        'epoch': FLAGS.epoch
    }
    
    Coresponding_time = '2018-12-14 10:59:10.650649'
    FLAGS.eval_log_dir = FLAGS.eval_log_dir+Coresponding_time
    checkpoint_dir = FLAGS.checkpoint_dir+Coresponding_time
    
    train_eval_log_dir = os.path.join(FLAGS.eval_log_dir, 'train')
    val_eval_log_dir = os.path.join(FLAGS.eval_log_dir, 'val')
    test_eval_log_dir = os.path.join(FLAGS.eval_log_dir, 'test')
    
    with open(tfrecords_meta_file, 'r') as f:
        content = json.load(f)
        num_train_examples = content['num_examples']['train']
        num_val_examples = content['num_examples']['val']
        num_test_examples = content['num_examples']['test']

    #training
    train(train_tfrecords_file, num_train_examples,
           val_tfrecords_file, num_val_examples,
           LogTrain_dir, restored_checkpoint,
           training_param, model_param)
    print('training ends, do a final evaluation') 

    #final evaluation on accuracy
    eval_total(checkpoint_dir, train_tfrecords_file, num_train_examples, train_eval_log_dir, model_param)
    eval_total(checkpoint_dir, val_tfrecords_file, num_val_examples, val_eval_log_dir, model_param)
    eval_total(checkpoint_dir, test_tfrecords_file, num_test_examples, test_eval_log_dir, model_param)


if __name__ == '__main__':
    tf.app.run(main=main)
