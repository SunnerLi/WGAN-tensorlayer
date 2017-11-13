from _init_path import *
from data_helper import ImageHandler, generateNoice
from model.model import WassersterinGAN, DCGAN
from record import saveGeneratedBatch
from train import trainGAN
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    handler = ImageHandler(dataset_name='lsun', resize_length=64)
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])

    # Train toward WGAN
    net = WassersterinGAN(img_channel=3)
    net.build(noise_ph, image_ph)
    trainGAN(noise_ph, image_ph, net, handler, 
        output_dir='output/lsun/wgan',
        output_csv_name='lsun_wgan.csv')

    # Train toward DCGAN
    net = DCGAN(img_channel=3)
    net.build(noise_ph, image_ph)
    trainGAN(noise_ph, image_ph, net, handler, 
        output_dir='output/lsun/dcgan',
        output_csv_name='lsun_dcgan.csv')