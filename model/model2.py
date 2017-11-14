from model.gan import GAN
import tensorlayer as tl
import tensorflow as tf
import math

"""
y=1-e^(-((   x-0    )/e)^2)
"""

class GaussianGAN1(GAN):
    def __init__(self, img_channel, filter_base = 32, fc_unit_num = 1024, conv_depth = 2, lambda_panelty_factor = 10.0, name='ggan1_'):
        super(GaussianGAN1, self).__init__(img_channel, filter_base, fc_unit_num, conv_depth, lambda_panelty_factor, name)

    def build(self, noise_ph, image_ph):
        # Update parameter and build network
        self.img_height = image_ph.get_shape().as_list()[1]
        self.img_width  = image_ph.get_shape().as_list()[2]
        self.buildGraph(noise_ph, image_ph)

        # Define origin loss (Earth Mover distance)
        self.generator_loss = tf.reduce_mean(self.fake_logits)
        self.discriminator_loss = tf.reduce_mean(self.true_logits) - tf.reduce_mean(self.fake_logits)        

        # Gradient panelty (Just regularization)
        for variable in self.variable_set:
            self.discriminator_loss += tf.contrib.layers.l2_regularizer(1e-8)(variable)

        # Define optimizer in order
        self.generator_optimize = None
        self.discriminator_optimize = None
        generator_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        discriminator_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.generator_optimize = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(self.generator_loss, var_list=generator_vars)
            self.discriminator_optimize = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5, beta2=0.9).minimize(self.discriminator_loss, var_list=discriminator_vars)        

class GaussianGAN2(GAN):
    def __init__(self, img_channel, filter_base = 32, fc_unit_num = 1024, conv_depth = 2, lambda_panelty_factor = 10.0, name='ggan2_'):
        super(GaussianGAN2, self).__init__(img_channel, filter_base, fc_unit_num, conv_depth, lambda_panelty_factor, name)

    def build(self, noise_ph, image_ph):
        # Update parameter and build network
        self.img_height = image_ph.get_shape().as_list()[1]
        self.img_width  = image_ph.get_shape().as_list()[2]
        self.buildGraph(noise_ph, image_ph)

        # Define origin loss (Earth Mover distance)
        self.generator_loss = tf.reduce_mean(self.fake_logits)
        self.discriminator_loss = tf.reduce_mean(self.true_logits) - tf.reduce_mean(self.fake_logits)

        # Gradient panelty (Revise gaussian distance panelty)
        epsilon = tf.random_normal([])
        combination_sample = epsilon * self.generated_tensor + (1 - epsilon) * image_ph
        self.buildPrint('Build panelty discriminator ...')
        panelty_logits = self.getDiscriminator(combination_sample, reuse=True)
        combination_gradient = tf.gradients(panelty_logits, combination_sample)[0] + 1e-8
        self.panelty_loss = tf.reduce_mean(1 - tf.exp(-tf.square(combination_gradient/math.e)))
        self.panelty_loss = tf.reduce_mean(tf.square(self.panelty_loss - 1.0) * self.lambda_panelty_factor)
        self.discriminator_loss += self.panelty_loss

        # Define optimizer in order
        self.generator_optimize = None
        self.discriminator_optimize = None
        generator_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        discriminator_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.generator_optimize = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(self.generator_loss, var_list=generator_vars)
            self.discriminator_optimize = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5, beta2=0.9).minimize(self.discriminator_loss, var_list=discriminator_vars)

class GaussianGAN3(GAN):
    def __init__(self, img_channel, filter_base = 32, fc_unit_num = 1024, conv_depth = 2, lambda_panelty_factor = 10.0, name='ggan3_'):
        super(GaussianGAN3, self).__init__(img_channel, filter_base, fc_unit_num, conv_depth, lambda_panelty_factor, name)

    def build(self, noise_ph, image_ph):
        # Update parameter and build network
        self.img_height = image_ph.get_shape().as_list()[1]
        self.img_width  = image_ph.get_shape().as_list()[2]
        self.buildGraph(noise_ph, image_ph)

        # Define origin loss (Earth Mover distance)
        self.generator_loss = tf.reduce_mean(self.fake_logits)
        self.discriminator_loss = tf.reduce_mean(self.true_logits) - tf.reduce_mean(self.fake_logits)

        # Gradient panelty (Merge the panelty discriminator)
        epsilon = tf.random_normal([])
        combination_gradient = epsilon * tf.gradients(self.fake_logits, self.generated_tensor)[0] + (1 - epsilon) * tf.gradients(self.true_logits, image_ph)[0]
        self.panelty_loss = tf.reduce_mean(1 - tf.exp(-tf.square(combination_gradient/math.e)))
        self.panelty_loss = tf.reduce_mean(tf.square(self.panelty_loss - 1.0) * self.lambda_panelty_factor)
        self.discriminator_loss += self.panelty_loss

        # Define optimizer in order
        self.generator_optimize = None
        self.discriminator_optimize = None
        generator_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        discriminator_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.generator_optimize = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(self.generator_loss, var_list=generator_vars)
            self.discriminator_optimize = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5, beta2=0.9).minimize(self.discriminator_loss, var_list=discriminator_vars)        

if __name__ == '__main__':
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = GaussianGAN3(img_channel=1)
    net.build(noise_ph, image_ph)