import tensorlayer as tl
import tensorflow as tf

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class GAN(object):
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
                channel = self.filter_base * (2 ** i)
                network = tl.layers.Conv2d(network, n_filter = channel, filter_size=(4, 4), strides=(2, 2), name ='discriminator_conv2d_%s'%str(i))
                network = tl.layers.BatchNormLayer(network, act = leaky_relu, name ='discriminator_batchnorm_layer_%s'%str(i), is_train = True)
            network = tl.layers.FlattenLayer(network)
            network = tl.layers.DenseLayer(network, n_units = 1, act = tf.identity, name = 'discriminator_dense_layer_final')
            return network.outputs

    def getGenerator(self, noise_ph):
        dense_recover_length = self.img_length // (2 ** self.conv_depth)
        print(dense_recover_length)
        if dense_recover_length * (2 ** self.conv_depth) != self.img_length:
            raise Exception('Invalid image length...')
        with tf.variable_scope(self.name + 'generator'):
            network = tl.layers.InputLayer(noise_ph)
            network = tl.layers.DenseLayer(network, n_units = self.fc_unit_num, name ='generator_dense_layer_1')
            network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_1', is_train = True)
            network = tl.layers.DenseLayer(network, n_units = dense_recover_length * dense_recover_length * self.filter_base * (2  ** (self.conv_depth - 1)), name ='generator_dense_layer_2')
            network = tl.layers.ReshapeLayer(network, tf.stack([tf.shape(noise_ph)[0], dense_recover_length, dense_recover_length, self.filter_base * (2  ** (self.conv_depth - 1))]))
            network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_2', is_train = True)
            print(network.outputs.get_shape())
            for i in range(self.conv_depth, 1, -1):
                height = dense_recover_length * (2 ** (self.conv_depth - i + 1))
                width  = dense_recover_length * (2 ** (self.conv_depth - i + 1))
                channel = self.filter_base * (2  ** (i - 2))
                network = tl.layers.DeConv2d(network, n_out_channel = channel, strides = (2, 2), filter_size=(4, 4), out_size = (height, width), name ='generator_decnn2d_%s'%str(self.conv_depth-i))
                network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_%s'%str(self.conv_depth-i+3), is_train = True)
            network = tl.layers.DeConv2d(network, n_out_channel = self.img_depth, strides = (2, 2), out_size = (self.img_height, self.img_width), name ='generator_decnn2d_final')
            network = tl.layers.ReshapeLayer(network, [tf.shape(noise_ph)[0], self.img_height, self.img_width, self.img_depth], name ='generator_reshape_layer')
            network = tl.layers.BatchNormLayer(network, act = tf.nn.tanh, name ='generator_batchnorm_layer_final', is_train = True)
            return network.outputs