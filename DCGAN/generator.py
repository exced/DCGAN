#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import model

import tensorflow as tf

class Generator:
    ''' generator neural network
    '''
    def __init__(self):
        # layers
        self.G_W1 = model.weight_variable([100, 128], name='G_W1')
        self.G_b1 = model.bias_variable([128], name='G_b1')

        self.G_W2 = model.weight_variable([128, 784], name='G_W2')
        self.G_b2 = model.bias_variable([784], name='G_b2')

        self.theta = [
            self.G_W1, self.G_b1,
            self.G_W2, self.G_b2,
        ]

    def generate(self, z):
        ''' generate new image
        '''
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        H = tf.matmul(G_h1, self.G_W2) + self.G_b2
        prob = tf.nn.sigmoid(H)
        return prob
