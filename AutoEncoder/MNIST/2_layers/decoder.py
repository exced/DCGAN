#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import model

import tensorflow as tf

class Decoder:
    ''' decoder neural network
    '''
    def __init__(self):
        # layers
        self.D_W1 = model.weight_variable([128, 256], name='D_W1')
        self.D_b1 = model.bias_variable([256], name='D_b1')

        self.D_W2 = model.weight_variable([256, 784], name='D_W2')
        self.D_b2 = model.bias_variable([784], name='D_b2')

        self.theta = [
            self.D_W1, self.D_b1,
            self.D_W2, self.D_b2,
        ]

    def decode(self, X):
        ''' decode
        '''
        D_h1 = tf.nn.sigmoid(tf.matmul(X, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.sigmoid(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        return D_h2
