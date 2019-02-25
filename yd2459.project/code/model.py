import tensorflow as tf
import numpy as np


class Model(object):
    
    @staticmethod
    def inference(x, conv_parameter, drop_rate):
        para_filter = conv_parameter['filters_num']
        para_kernal = conv_parameter['kernel_size']
        para_pool = conv_parameter['pool_size']
        para_stride = conv_parameter['strides']
        para_dense = conv_parameter['dense']
        para_flatten = conv_parameter['flatten']

        Clayer_Num = len(para_filter)
        Dlayer_Num = len(para_dense)

        conv_in = []
        conv_in.append(x)
        for conv_cnt in range(Clayer_Num):
            layer_name = 'hidden'+str(conv_cnt+1)
            with tf.variable_scope(layer_name):
                conv = tf.layers.conv2d(conv_in[conv_cnt],filters=para_filter[conv_cnt],kernel_size=[para_kernal[conv_cnt],para_kernal[conv_cnt]], padding='same')
                norm = tf.layers.batch_normalization(conv)
                activation = tf.nn.relu(norm)
                pool = tf.layers.max_pooling2d(activation, pool_size=[para_pool[conv_cnt],para_pool[conv_cnt]],strides=para_stride[conv_cnt], padding='same')
                dropout = tf.layers.dropout(pool, rate=drop_rate)
                conv_in.append(dropout)
        
        flatten = tf.reshape(conv_in[-1], [-1, para_flatten[0] * para_flatten[1] * para_filter[-1]])

        dense_in = []
        dense_in.append(flatten)
        for dense_cnt in range(Dlayer_Num):
            layer_name = 'hidden'+str(Clayer_Num+dense_cnt+1)
            with tf.variable_scope(layer_name):
                dense = tf.layers.dense(dense_in[dense_cnt], units=para_dense[dense_cnt], activation=tf.nn.relu)
                dense_in.append(dense)

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(dense_in[-1], units=7)
            length = dense
        
        digit_pred = []
        for digit_cnt in range(5):
            layer_name = 'digit'+str(digit_cnt+1)
            with tf.variable_scope(layer_name):
                dense = tf.layers.dense(dense_in[-1], units=11)
                digit_pred.append(dense)

        length_logits, digits_logits = length, tf.stack([digit_pred[0],digit_pred[1],digit_pred[2],digit_pred[3],digit_pred[4]], axis=1) 
        return length_logits, digits_logits

    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy+digit1_cross_entropy+digit2_cross_entropy+digit3_cross_entropy+digit4_cross_entropy+digit5_cross_entropy 
        return loss
