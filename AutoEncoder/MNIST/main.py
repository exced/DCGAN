#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math
import argparse
import sys

import model
import encoder
import decoder

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# Hyper parameters
learning_rate = 0.002
batch_size = 256
training_iter = 50
display_step = 1
nb_test = 5

def plot(datas, mnist):
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist[i], (28, 28)))
        a[1][i].imshow(np.reshape(datas[i], (28, 28)))
    return f

def save(i, samples, mnist):
    fig = plot(samples, mnist)
    plt.savefig((FLAGS.out_dir + '{}/out.png').format(str(i).zfill(3)), bbox_inches='tight')
    plt.close(fig)

def main(_):
    # create foler for generated images
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # data placeholders
    X = tf.placeholder(tf.float32, shape=[None, 784])

    # initializes encoder and decoder
    _encoder = encoder.Encoder()
    _decoder = decoder.Decoder()

    # encodes and decodes
    X_fake = _decoder.decode(_encoder.encode(X))

    # loss : quadratic error
    loss = tf.reduce_mean(tf.pow(X - X_fake, 2))

    # invoke the optimizer
    solver = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nb_batch = int(mnist.train.num_examples/batch_size)
        # train
        for it in range(training_iter):
            for i in range(nb_batch):
                # next input batch
                X_mb, _ = mnist.train.next_batch(batch_size)

                # run session
                _, loss_curr = sess.run([solver, loss], feed_dict={X: X_mb})

            if it % display_step == 0:
                print('Iter: {}'.format(it))
                print('cost: {:.4}'. format(loss_curr))

        # test
        for it in range(10*nb_test):
            # next test batch
            T_mb, _ = mnist.test.next_batch(10)
            test = sess.run(X_fake, feed_dict={X: T_mb})
            save(it, test, T_mb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../MNIST_data', help='Directory for storing input data')
    parser.add_argument('--out_dir', type=str, default='./out', help='Directory for storing output data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
