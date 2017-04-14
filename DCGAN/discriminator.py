#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import model

import tensorflow as tf

class Discriminator:
    ''' discriminator neural network
    '''

    def __init__(self):
        self.D_W1 = model.weight_variable([5, 5, 1, 16], name='D_W1')
        self.D_b1 = model.bias_variable([16], name='D_b1')

        self.D_W2 = model.weight_variable([3, 3, 16, 32], name='D_W2')
        self.D_b2 = model.bias_variable([32], name='D_b2')

        self.D_W3 = model.weight_variable([7 * 7 * 32, 128], name='D_W3')
        self.D_b3 = model.bias_variable([128], name='D_b3')

        self.D_W4 = model.weight_variable([128, 1], name='D_W4')
        self.D_b4 = model.bias_variable([1], name='D_b4')

        self.theta = [
            self.D_W1, self.D_b1,
            self.D_W2, self.D_b2,
            self.D_W3, self.D_b3,
            self.D_W4, self.D_b4,
        ]

    def discriminate(self, X):
        ''' discriminator graph
        '''
        # logic    
        D = tf.reshape(X, shape=[-1, 28, 28, 1])
        D_H1 = model.leaky_relu(tf.nn.conv2d(D, self.D_W1, strides=[1, 2, 2, 1], padding='SAME') + self.D_b1)
        D_H2 = model.leaky_relu(tf.nn.conv2d(D_H1, self.D_W2, strides=[1, 2, 2, 1], padding='SAME') + self.D_b2)
        D_H2 = tf.reshape(D_H2, shape=[-1, 7 * 7 * 32])
        H = model.leaky_relu(tf.matmul(D_H2, self.D_W3) + self.D_b3)
        logit = tf.matmul(H, self.D_W4) + self.D_b4
        prob = tf.nn.sigmoid(logit)
        return prob, logit
