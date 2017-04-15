#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import math
import argparse
import sys

import model
import encoder
import decoder
import utils

import tensorflow as tf

FLAGS = None

# Hyper parameters
learning_rate = 0.0002
batch_size = 128
training_iter = 100
display_step = 1

def main(_):
    # create foler for generated images
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    # Import data
    datas = filter(lambda x: x.endswith('jpg'), os.listdir(FLAGS.data_dir))

    # data placeholders
    X = tf.placeholder(tf.float32, shape=[None, 4096]) # 64*64 = 4096

    # initializes encoder and decoder
    _encoder = encoder.Encoder(pkeep=0.75)
    _decoder = decoder.Decoder(pkeep=0.75)

    # encodes and decodes
    X_fake = _decoder.decode(_encoder.encode(X))

    # loss : quadratic error
    loss = tf.reduce_mean(tf.pow(X - X_fake, 2))

    # invoke the optimizer
    solver = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train
        for it in range(training_iter):
            for start, end in zip(range(0, len(datas), batch_size), range(batch_size, len(datas), batch_size)):
                # next input batch
                batch_image_files = datas[start:end]
                batch_images = map(lambda x: utils.crop_resize( os.path.join( FLAGS.data_dir, x) ), batch_image_files)
                batch_images = np.array(batch_images).astype(np.float32)

            # keep the last batch for tesing
            if it == training_iter-1:

                # run session
                _, loss_curr = sess.run([solver, loss], feed_dict={X: batch_images})

            if it % display_step == 0:
                print('Iter: {}'.format(it))
                print('cost: {:.4}'. format(loss_curr))

            else:
                # test on 10 test images
                test = sess.run(X_fake, feed_dict={X: batch_images[:10]})
                utils.save(test, batch_images[:10])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../../data/celebA/', help='Directory for storing input data')
    parser.add_argument('--out_dir', type=str, default='./out', help='Directory for storing output data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
