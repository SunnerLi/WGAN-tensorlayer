from _init_path import *
from model2 import GaussianGAN1, GaussianGAN2, GaussianGAN3
from data_helper import ImageHandler, generateNoice
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 64, 64, 1])
    net = GaussianGAN3(img_channel=1)
    net.build(noise_ph, image_ph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1):
            _ = sess.run([net.discriminator_optimize], feed_dict={
                noise_ph: generateNoice(batch_size=32, dim=100),
                image_ph: np.ones([32, 64, 64, 1])
            })
            _ = sess.run([net.generator_optimize], feed_dict={
                noise_ph: generateNoice(batch_size=32, dim=100),
                image_ph: np.ones([32, 64, 64, 1])
            })
        