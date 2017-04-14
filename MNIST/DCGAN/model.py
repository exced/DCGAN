#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import math

import tensorflow as tf

def lrelu(x, leak=0.2):
    '''leakyRelu activation function
    '''
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


def weight_variable(shape, name, stddev=0.02):
    '''weight_variable generates a weight variable of a given shape.
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name))


def bias_variable(shape, name):
    '''bias_variable generates a bias variable of a given shape.
    '''
    return tf.Variable(tf.constant(0.1, shape=shape, name=name))


def conv2d(x, W):
    '''conv2d returns a 2d convolution layer with full stride.
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class DCGAN():
    '''Deep Convolutional Generative Adversarial Network
    '''
    def __init__(self,
                 batch_size=100,
                 image_shape=[28, 28, 1],
                 dim_z=100,
                 dim_y=10,
                 weights={},
                 biases={},
                 ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.weights = {}
        # add weights variables
        for key, value in weights.items():
            self.weights[key] = weight_variable(shape=value, name=key)

        # variables of discriminator and generator
        self.vars_discriminator = list(filter(lambda x: x.name.startswith('discriminator'), tf.trainable_variables()))
        self.vars_generator = list(filter(lambda x: x.name.startswith('generator'), tf.trainable_variables()))

        self.biases = {}
        # add biases variables
        for key, value in biases.items():
            self.biases[key] = bias_variable(shape=value, name=key)

        fully_connected_size = image_shape[0]*image_shape[1] # 28 * 28 = 784
        self.X = tf.placeholder(tf.float32, shape=[None, fully_connected_size]) # 784
        self.Z = tf.placeholder(tf.float32, shape=[None, dim_z]) # 100


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def dc_generator(z):
    pass


def dc_discriminator(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(tf.nn.conv2d(x, DC_D_W1, strides=[1, 2, 2, 1], padding='SAME') + DC_D_b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, DC_D_W2, strides=[1, 2, 2, 1], padding='SAME') + DC_D_b2)
    conv2 = tf.reshape(conv2, shape=[-1, 7 * 7 * 32])
    h = tf.nn.relu(tf.matmul(conv2, DC_D_W3) + DC_D_b3)
    logit = tf.matmul(h, DC_D_W4) + DC_D_b4
    prob = tf.nn.sigmoid(logit)

    return prob, logit

def main(_):
    # create foler for generated images
    if not os.path.exists('./out/'):
        os.makedirs('./out/')

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # initializes generator and discriminator
    G_sample = generator(Z)
    D_real, D_logit_real = dc_discriminator(X)
    D_fake, D_logit_fake = dc_discriminator(G_sample)

    # losses:
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        D_logit_real, tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        D_logit_fake, tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        D_logit_fake, tf.ones_like(D_logit_fake)))

    # invoke the optimizers
    D_solver = tf.train.AdamOptimizer(
        0.0001).minimize(D_loss, var_list=theta_DC_D)
    G_solver = tf.train.AdamOptimizer(
        0.0001).minimize(G_loss, var_list=theta_G)

    mb_size = 128
    Z_dim = 100

    # image number id
    i = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for it in range(10000):
            if it % 100 == 0:
                samples = sess.run(G_sample, feed_dict={
                                   Z: sample_Z(16, Z_dim)})

                fig = plot(samples)
                plt.savefig('./out/{}.png'.format(str(i).zfill(3)),
                            bbox_inches='tight')
                i += 1
                plt.close(fig)

            X_mb, _ = mnist.train.next_batch(mb_size)

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                X: X_mb,
                Z: sample_Z(mb_size, Z_dim)
            })
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                Z: sample_Z(mb_size, Z_dim)
            })

            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
