from _init_path import *
from data_helper import ImageHandler, generateNoice
from model.model import WassersterinGAN, DCGAN
from record import saveGeneratedBatch
from train import trainGAN
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
    handler = ImageHandler(dataset_name='mnist', resize_length=32)
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 32, 32, 1])
    net = DCGAN(img_channel=1)
    net.build(noise_ph, image_ph)
    trainGAN(noise_ph, image_ph, net, handler, 
        output_dir='output/mnist/dcgan',
        output_csv_name='mnist_dcgan.csv')