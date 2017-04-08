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

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# variable learning rate
lr = tf.placeholder(tf.float32)
# dropout probability
pkeep = tf.placeholder(tf.float32)


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = weight_variable([784, 128])
D_b1 = bias_variable([128])

D_W2 = weight_variable([128, 1])
D_b2 = bias_variable([1])

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = weight_variable([100, 128])
G_b1 = bias_variable([128])

G_W2 = weight_variable([128, 784])
G_b2 = bias_variable([784])

theta_G = [G_W1, G_W2, G_b1, G_b2]


DC_D_W1 = weight_variable([5, 5, 1, 16])
DC_D_b1 = bias_variable([16])

DC_D_W2 = weight_variable([3, 3, 16, 32])
DC_D_b2 = bias_variable([32])

DC_D_W3 = weight_variable([7 * 7 * 32, 128])
DC_D_b3 = bias_variable([128])

DC_D_W4 = weight_variable([128, 1])
DC_D_b4 = bias_variable([1])

theta_DC_D = [DC_D_W1, DC_D_b1, DC_D_W2,
              DC_D_b2, DC_D_W3, DC_D_b3, DC_D_W4, DC_D_b4]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


def dc_generator(z):
    pass


def dc_discriminator(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(tf.nn.conv2d(x, DC_D_W1, strides=[
                       1, 2, 2, 1], padding='SAME') + DC_D_b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, DC_D_W2, strides=[
                       1, 2, 2, 1], padding='SAME') + DC_D_b2)
    conv2 = tf.reshape(conv2, shape=[-1, 7 * 7 * 32])
    h = tf.nn.relu(tf.matmul(conv2, DC_D_W3) + DC_D_b3)
    logit = tf.matmul(h, DC_D_W4) + DC_D_b4
    prob = tf.nn.sigmoid(logit)

    return prob, logit


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
        0.005).minimize(D_loss, var_list=theta_DC_D)
    G_solver = tf.train.AdamOptimizer(0.005).minimize(G_loss, var_list=theta_G)

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
                                      X: X_mb, Z: sample_Z(mb_size, Z_dim)})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                                      Z: sample_Z(mb_size, Z_dim)})

            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='../MNIST_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
