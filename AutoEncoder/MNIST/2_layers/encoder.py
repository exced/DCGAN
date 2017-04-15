#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import model

import tensorflow as tf

class Encoder:
    ''' Encoder neural network
    '''

    def __init__(self):
        # layers
        self.E_W1 = model.weight_variable([784, 256], name='E_W1')
        self.E_b1 = model.bias_variable([256], name='E_b1')

        self.E_W2 = model.weight_variable([256, 128], name='E_W2')
        self.E_b2 = model.bias_variable([128], name='E_b2')

        self.theta = [
            self.E_W1, self.E_b1,
            self.E_W2, self.E_b2,
        ]

    def encode(self, X):
        ''' encode
        '''
        E_h1 = tf.nn.sigmoid(tf.matmul(X, self.E_W1) + self.E_b1)
        E_h2 = tf.nn.sigmoid(tf.matmul(E_h1, self.E_W2) + self.E_b2)
        return E_h2
