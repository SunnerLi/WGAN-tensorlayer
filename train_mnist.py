from _init_path import *
from data_helper import ImageHandler, generateNoice
from record import saveGeneratedBatch
from wgan import WassersterinGAN
from train import trainGAN
from dcgan import DCGAN
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # Train toward WGAN    
    handler = ImageHandler(dataset_name='mnist', resize_length=28)
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = WassersterinGAN(img_channel=1)
    net.build(noise_ph, image_ph)
    trainGAN(noise_ph, image_ph, net, handler, 
        output_dir='output/mnist/wgan',
        output_csv_name='mnist_wgan.csv')

    # Train toward DCGAN
    # handler = ImageHandler(dataset_name='mnist', resize_length=64)
    # noise_ph = tf.placeholder(tf.float32, [None, 100])
    # image_ph = tf.placeholder(tf.float32, [None, 64, 64, 1])
    # net = DCGAN(img_channel=1)
    # net.build(noise_ph, image_ph)
    # trainGAN(noise_ph, image_ph, net, handler, 
        # output_dir='output/mnist/dcgan',
        # output_csv_name='mnist_dcgan.csv')