from data_helper import ImageHandler, generateNoice
from model import WassersterinGAN, DCGAN
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
    net = WassersterinGAN(img_channel=3)
    net.build(noise_ph, image_ph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1):
            _ = sess.run([net.discriminator_optimize], feed_dict={
                noise_ph: generateNoice(batch_size=32, dim=100),
                image_ph: np.ones([32, 64, 64, 3])
            })
            _ = sess.run([net.generator_optimize], feed_dict={
                noise_ph: generateNoice(batch_size=32, dim=100),
                image_ph: np.ones([32, 64, 64, 3])
            })
        