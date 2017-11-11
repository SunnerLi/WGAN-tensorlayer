from gan import GAN
import tensorlayer as tl
import tensorflow as tf

class WassersterinGAN(GAN):
    def __init__(self, img_channel, filter_base = 16, fc_unit_num=1024, conv_depth=2, lambda_panelty_factor = 10.0):
        super(WassersterinGAN, self).__init__(filter_base, fc_unit_num, conv_depth, lambda_panelty_factor, img_channel)

    def build(self, noise_ph, image_ph):
        # Update parameter and build network
        self.img_height = image_ph.get_shape().as_list()[1]
        self.img_width  = image_ph.get_shape().as_list()[2]
        self.buildGraph(noise_ph, image_ph)

        # Define origin loss (Earth Mover distance)
        self.generator_loss = tf.reduce_mean(self.fake_logits)
        self.discriminator_loss = tf.reduce_mean(self.true_logits) - tf.reduce_mean(self.fake_logits)

        # Gradient panelty
        epsilon = tf.random_normal([])
        combination_sample = epsilon * self.generated_tensor + (1 - epsilon) * image_ph
        panelty_logits = self.getDiscriminator(combination_sample, reuse=True)
        combination_gradient = tf.gradients(panelty_logits, combination_sample)[0]
        self.panelty_loss = tf.sqrt(tf.reduce_mean(tf.square(combination_gradient), axis=1))
        self.panelty_loss = tf.reduce_mean(tf.square(self.panelty_loss - 1) * self.lambda_panelty_factor)
        self.discriminator_loss += self.panelty_loss

        # Define optimizer in order
        self.generator_optimize = None
        self.discriminator_optimize = None
        generator_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        discriminator_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.generator_optimize = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(self.generator_loss, var_list=generator_vars)
            self.discriminator_optimize = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(self.discriminator_loss, var_list=discriminator_vars)


class DCGAN(GAN):
    def __init__(self, img_channel, filter_base = 32, fc_unit_num=1024, conv_depth=2, lambda_panelty_factor = 10.0):
        super(DCGAN, self).__init__(filter_base, fc_unit_num, conv_depth, lambda_panelty_factor, img_channel)

    def build(self, noise_ph, image_ph):
        # Update parameter and build network
        self.img_height = image_ph.get_shape().as_list()[1]
        self.img_width  = image_ph.get_shape().as_list()[2]
        self.buildGraph(noise_ph, image_ph)

        # Define origin loss (Jensen Shannon divergence)
        self.generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.ones_like(self.fake_logits))
        self.discriminator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits)) + \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logits, labels=tf.ones_like(self.true_logits))

        # Define optimizer in order
        self.generator_optimize = None
        self.discriminator_optimize = None
        generator_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        discriminator_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.generator_optimize = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(self.generator_loss, var_list=generator_vars)
            self.discriminator_optimize = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(self.discriminator_loss, var_list=discriminator_vars)


if __name__ == '__main__':
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = DCGAN()
    net.build(noise_ph, image_ph)
