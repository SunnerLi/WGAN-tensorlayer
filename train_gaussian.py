from _init_path import *
from model.model2 import GaussianGAN1, GaussianGAN2, GaussianGAN3
from data_helper import ImageHandler, generateNoice
from record import saveGeneratedBatch
from train import trainGAN
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    handler = ImageHandler(dataset_name='mnist', resize_length=28)
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = GaussianGAN1(img_channel=1)
    net.build(noise_ph, image_ph)
    trainGAN(noise_ph, image_ph, net, handler, 
        output_dir='output/mnist/gaussian1',
        output_csv_name='mnist_gaussian1.csv')

    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = GaussianGAN2(img_channel=1)
    net.build(noise_ph, image_ph)
    trainGAN(noise_ph, image_ph, net, handler, 
        output_dir='output/mnist/gaussian2',
        output_csv_name='mnist_gaussian2.csv')

    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = GaussianGAN3(img_channel=1)
    net.build(noise_ph, image_ph)
    trainGAN(noise_ph, image_ph, net, handler, 
        output_dir='output/mnist/gaussian3',
        output_csv_name='mnist_gaussian3.csv')