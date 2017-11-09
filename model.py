import tensorlayer as tl
import tensorflow as tf

class WGAN(object):
    def __init__(self, filter_base = 2, fc_unit_num=1024, conv_depth=2):
        self.filter_base = filter_base
        self.fc_unit_num = fc_unit_num
        self.conv_depth = conv_depth

    def build(self, noise_ph, image_ph):
        generated_tensor = self.getGenerator(noise_ph)
        print('\n\n')
        self.true_logits = self.getDiscriminator(image_ph, reuse=False)
        print('\n\n')
        self.fake_logits = self.getDiscriminator(generated_tensor, reuse=True)

    def getDiscriminator(self, img_ph, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.InputLayer(img_ph, name ='discriminator_input_layer')
            for i in range(self.conv_depth):
                network = tl.layers.Conv2d(network, n_filter = self.filter_base * (i + 1), name ='discriminator_conv2d_%s'%str(1+i*2))
                network = tl.layers.Conv2d(network, n_filter = self.filter_base * (i + 1), name ='discriminator_conv2d_%s'%str(2+i*2))
                network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='discriminator_batchnorm_layer_%s'%str(i))
                network = tl.layers.MaxPool2d(network, name='discriminator_maxpool_%s'%str(i))
            print('shape: ', network.outputs.get_shape())
            network = tl.layers.FlattenLayer(network)
            network = tl.layers.DenseLayer(network, n_units = self.fc_unit_num, act = tf.nn.relu, name ='discriminator_dense_layer')
            network = tl.layers.DenseLayer(network, n_units = 1)
            return network.outputs

    def getGenerator(self, noise_ph):
        with tf.variable_scope('generator'):
            network = tl.layers.InputLayer(noise_ph)
            network = tl.layers.DenseLayer(network, n_units = self.fc_unit_num, name ='generator_dense_layer_1')
            network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_1')
            network = tl.layers.DenseLayer(network, n_units = 7 * 7 * self.conv_depth * self.filter_base, name ='generator_dense_layer_2')
            network = tl.layers.ReshapeLayer(network, tf.stack([tf.shape(noise_ph)[0], 7, 7, self.conv_depth * self.filter_base]))
            network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_2')
            for i in range(self.conv_depth - 1, -1, -1):
                height = 7 * (2 ** (self.conv_depth-i))
                width  = 7 * (2 ** (self.conv_depth-i))
                if i == 0:
                    channel = 1
                else:
                    channel = self.filter_base * i
                print(height, width, channel)
                network = tl.layers.DeConv2d(network, n_out_channel = channel, strides = (2, 2), out_size = (height, width), name ='generator_decnn2d_%s'%str(1+i*2))
                print(channel)



                
                network = tl.layers.DeConv2d(network, n_out_channel = channel, strides = (2, 2), out_size = (height, width), name ='generator_decnn2d_%s'%str(2+i*2))
                network = tl.layers.BatchNormLayer(network, act = tf.nn.relu, name ='generator_batchnorm_layer_%s'%str(self.conv_depth-i+2))
            return network.outputs

if __name__ == '__main__':
    noise_ph = tf.placeholder(tf.float32, [None, 100])
    image_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = WGAN()
    # net.build(noise_ph, image_ph)
    net.getGenerator(noise_ph)