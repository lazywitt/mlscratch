'''
This code represents a nueral network for a json data i got from kaggle. Here i have only posted the Nueral network part of the overall program
'''

#source URL =url=https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/download/train.json.7z

import pandas as pd
import numpy as np
import tensorflow as tf
with tf.variable_scope('Convolutionalnetwork'):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    c1 = tf.layers.conv2d(X, filters=32,  kernel_size=[5, 5], activation=tf.nn.relu)
    p1 = tf.layers.max_pooling2d(c1, pool_size=[2, 2], strides=2)
    c2 = tf.layers.conv2d(pool1, filters=64,  kernel_size=[3,3], activation=tf.nn.relu)
    p2 = tf.layers.max_pooling2d(c2, pool_size=[2, 2], strides=2)
    c3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3,3], activation=tf.nn.relu)
    p3 = tf.layers.max_pooling2d(c3, pool_size=[2, 2], strides=2)
    FC_a = tf.contrib.layers.flatten(p3)
    FC_b = tf.layers.dense(FC_a, 32, 
                        kernel_initializer=he_init, activation=tf.nn.relu)
    FC_c = tf.layers.dropout(FC_b, rate=dropout)
    logits = tf.layers.dense(FC_c, num_classes, activation=tf.nn.sigmoid)

'''
this sample can now be trained using the Nueral network
'''
