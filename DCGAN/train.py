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
import generator
import discriminator

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# Hyper parameters
learning_rate = 0.0002 
batch_size = 128
Z_dim = 100
training_iter = 10000
display_step = 100

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def main(_):

    # create foler for generated images
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # data placeholders
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Z = tf.placeholder(tf.float32, shape=[None, 100])

    # initializes generator and discriminator
    _generator = generator.Generator()
    _discriminator = discriminator.Discriminator()
    
    G_sample = _generator.generate(Z)
    D_real, D_logit_real = _discriminator.discriminate(X)
    D_fake, D_logit_fake = _discriminator.discriminate(G_sample)

    # losses:
    D_loss_real = model.reduce_mean(D_logit_real, tf.ones_like(D_logit_real))
    D_loss_fake = model.reduce_mean(D_logit_fake, tf.zeros_like(D_logit_fake))
    D_loss = D_loss_real + D_loss_fake
    G_loss = model.reduce_mean(D_logit_fake, tf.ones_like(D_logit_fake))

    # invoke the optimizers
    D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=_discriminator.theta)
    G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=_generator.theta)

    # image number id
    i = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for it in range(training_iter):
            if it % display_step == 0:
                samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

                # plot
                fig = plot(samples)
                plt.savefig(FLAGS.out_dir + '{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
                i += 1
                plt.close(fig)
            
            # next input batch
            X_mb, _ = mnist.train.next_batch(batch_size)

            # run session
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

            if it % display_step == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MNIST_data', help='Directory for storing input data')
    parser.add_argument('--out_dir', type=str, default='./out', help='Directory for storing output data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)