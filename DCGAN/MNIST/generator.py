#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import model

import tensorflow as tf

class Generator:
    ''' generator neural network
    '''
    def __init__(self):
        # layers
        self.G_W1 = model.weight_variable([100, 1024], name='G_W1')
        self.G_b1 = model.bias_variable([1024], name='G_b1')

        self.G_W2 = model.weight_variable([1024, 128], name='G_W2')
        self.G_b2 = model.bias_variable([128], name='G_b2')

        self.G_W3 = model.weight_variable([128, 64], name='G_W3')
        self.G_b3 = model.bias_variable([64], name='G_b3')

        self.G_W4 = model.weight_variable([64, 1], name='G_W4')
        self.G_b4 = model.bias_variable([1], name='G_b4')

        self.theta = [
            self.G_W1, self.G_b1,
            self.G_W2, self.G_b2,
            self.G_W3, self.G_b3,
            self.G_W4, self.G_b4,
        ]

    def generate(self, z):
        ''' generate new image
        '''
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.relu(tf.matmul(z, self.G_W2) + self.G_b2)
        G_h3 = tf.nn.relu(tf.matmul(z, self.G_W3) + self.G_b3)
        H = tf.matmul(G_h3, self.G_W3) + self.G_b3
        prob = tf.nn.sigmoid(H)
        return prob
