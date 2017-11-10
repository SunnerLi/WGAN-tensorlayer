from model import WassersterinGAN
import tensorflow as tf

if __name__ == '__main__':
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = WassersterinGAN()
    net.build(noise_ph, image_ph)