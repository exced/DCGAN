#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

WEIGHTS = {
    'generator_h1': [100, 256],
    'generator_h2': [256, 512],
    'generator_h3': [512, 784],
    'disciminator_h1': [784, 256],
    'disciminator_h2': [256, 128],
    'disciminator_h3': [128, 1],
}
BIASES = {
    'generator_b1': [256],
    'generator_b2': [512],
    'generator_b3': [784],
    'disciminator_b1': [256],
    'disciminator_b2': [128],
    'disciminator_b3': [1],
}

dcgan = DCGAN()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='../MNIST_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)