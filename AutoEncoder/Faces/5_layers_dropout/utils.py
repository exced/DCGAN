#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def crop_resize(image_path, resize_shape=(64,64)):
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    if width == height:
        resized_image = cv2.resize(image, resize_shape)
    elif width > height:
        resized_image = cv2.resize(image, (int(width * float(resize_shape[0])/height), resize_shape[1]))
        cropping_length = int( (resized_image.shape[1] - resize_shape[0]) / 2)
        resized_image = resized_image[:,cropping_length:cropping_length+resize_shape[1]]
    else:
        resized_image = cv2.resize(image, (resize_shape[0], int(height * float(resize_shape[1])/width)))
        cropping_length = int( (resized_image.shape[0] - resize_shape[1]) / 2)
        resized_image = resized_image[cropping_length:cropping_length+resize_shape[0], :]

def plot(datas, mnist):
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist[i], (64, 64)))
        a[1][i].imshow(np.reshape(datas[i], (64, 64)))
    return f

def save(samples, mnist):
    fig = plot(samples, mnist)
    plt.savefig((FLAGS.out_dir + '/out.png').format(), bbox_inches='tight')
    plt.close(fig)