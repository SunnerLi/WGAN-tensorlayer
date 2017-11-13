import tensorlayer as tl
import tensorflow as tf

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class WassersterinGAN(object):
    def __init__(self, img_channel, filter_base = 16, fc_unit_num = 1024, conv_depth = 2, lambda_panelty_factor = 10.0, name='wgan_'):
        self.filter_base = filter_base
        self.fc_unit_num = fc_unit_num
        self.conv_depth = conv_depth
        self.lambda_panelty_factor = lambda_panelty_factor
        self.img_depth = img_channel
        self.name = name

    def buildPrint(self, string):
        print('-' * 100)
        print(string)
        print('-' * 100)

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
        self.buildPrint('Build panelty discriminator ...')
        panelty_logits = self.getDiscriminator(combination_sample, reuse=True)
        combination_gradient = tf.gradients(panelty_logits, combination_sample)[0] + 1e-8
        self.panelty_loss = tf.sqrt(tf.reduce_mean(tf.square(combination_gradient), axis=1))
        self.panelty_loss = tf.reduce_mean(tf.square(self.panelty_loss - 1.0) * self.lambda_panelty_factor)
        self.discriminator_loss += self.panelty_loss

        # Define optimizer in order
        self.generator_optimize = None
        self.discriminator_optimize = None
        generator_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        discriminator_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.generator_optimize = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(self.generator_loss, var_list=generator_vars)
            self.discriminator_optimize = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.5, beta2=0.9).minimize(self.discriminator_loss, var_list=discriminator_vars)

    def buildGraph(self, noise_ph, image_ph):
        self.img_length = image_ph.get_shape()[2]
        self.buildPrint('Build generator ...')
        self.generated_tensor = self.getGenerator(noise_ph)
        self.buildPrint('Build true discriminator ...')
        self.true_logits = self.getDiscriminator(image_ph, reuse=False)
        self.buildPrint('Build fake discriminator ...')
        self.fake_logits = self.getDiscriminator(self.generated_tensor, reuse=True)

    def getDiscriminator(self, img_ph, reuse=False):
        with tf.variable_scope(self.name + 'discriminator', reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.InputLayer(img_ph, name ='discriminator_input_layer')
            network = tl.layers.ReshapeLayer(network, [tf.shape(img_ph)[0], self.img_height, self.img_width, self.img_depth], name ='discriminator_reshape_layer')
            for i in range(self.conv_depth):
                network = tl.layers.Conv2d(network, n_filter = self.filter_base, name ='discriminator_conv2d_%s'%str(i))
                network = tl.layers.BatchNormLayer(network, act = leaky_relu, name ='discriminator_batchnorm_layer_%s'%str(i))
                # network = tl.layers.MaxPool2d(network, name='discriminator_maxpool_%s'%str(i))
            network = tl.layers.FlattenLayer(network)
            # network = tl.layers.DenseLayer(network, n_units = self.fc_unit_num, act = tf.nn.relu, name = 'discriminator_dense_layer')
            network = tl.layers.DenseLayer(network, n_units = 1, act = tf.identity, name = 'discriminator_dense_layer_final')
            # Only use identity in the last layer, or the collapse problem occures
            return network.outputs

    def getGenerator(self, noise_ph):
        dense_recover_length = self.img_length // (2 ** self.conv_depth)
        if dense_recover_length * (2 ** self.conv_depth) != self.img_length:
            raise Exception('Invalid image length...')
        with tf.variable_scope(self.name + 'generator'):
            network = tl.layers.InputLayer(noise_ph)
            network = tl.layers.DenseLayer(network, n_units = self.fc_unit_num, name ='generator_dense_layer_1')
            network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_1')
            network = tl.layers.DenseLayer(network, n_units = dense_recover_length * dense_recover_length * self.filter_base * (2  ** (self.conv_depth - 1)), name ='generator_dense_layer_2')
            network = tl.layers.ReshapeLayer(network, tf.stack([tf.shape(noise_ph)[0], dense_recover_length, dense_recover_length, self.filter_base * (2  ** (self.conv_depth - 1))]))
            network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_2')
            print(network.outputs.get_shape())
            for i in range(self.conv_depth, 1, -1):
                height = dense_recover_length * (2 ** (self.conv_depth - i + 1))
                width  = dense_recover_length * (2 ** (self.conv_depth - i + 1))
                channel = self.filter_base * (2  ** (i - 1))
                network = tl.layers.DeConv2d(network, n_out_channel = channel, strides = (2, 2), out_size = (height, width), name ='generator_decnn2d_%s'%str(1+i*2))
                network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_%s'%str(self.conv_depth-i+3))
            network = tl.layers.DeConv2d(network, n_out_channel = self.img_depth, strides = (2, 2), out_size = (self.img_height, self.img_width), name ='generator_decnn2d_final')
            network = tl.layers.ReshapeLayer(network, [tf.shape(noise_ph)[0], self.img_height, self.img_width, self.img_depth], name ='generator_reshape_layer')
            network = tl.layers.BatchNormLayer(network, act = tf.nn.tanh, name ='generator_batchnorm_layer_final')
            return network.outputs

if __name__ == '__main__':
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = DCGAN(img_channel=1)
    net.build(noise_ph, image_ph)