#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def weight_variable(shape, name, stddev=0.1):
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

def leaky_relu(X, leak=0.2):
    '''leaky_relu activation function
    '''
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def reduce_mean(logits, labels):
    '''reduce with sigmoid cross entropy with logits
    '''
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    