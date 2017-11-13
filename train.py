from data_helper import ImageHandler, generateNoice
from model import WassersterinGAN, DCGAN
from visualize import saveGeneratedBatch
import tensorflow as tf
import pandas as pd
import numpy as np

epoch = 500

def trainGAN(noise_ph, image_ph, net, image_handler, output_img_dir, output_csv_dir, batch_size=32, n_critic=5):
    generator_loss_list = []
    discriminator_loss_list = []

    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                _generator_loss_list = []
                _discriminator_loss_list = []
                for j in range(image_handler.getImageNum() // batch_size):
                    if type(net) == WassersterinGAN:
                        discriminator_loss = 0.0
                        for k in range(n_critic):
                            feed_dict = {
                                noise_ph: generateNoice(batch_size, 100),
                                image_ph: image_handler.getBatchImage()
                            }
                            _discriminator_loss, _ = sess.run([net.discriminator_loss, net.discriminator_optimize], feed_dict=feed_dict)
                            discriminator_loss += _discriminator_loss
                        _discriminator_loss_list.append(discriminator_loss / 5)
                    elif type(net) == DCGAN:
                        feed_dict = {
                            noise_ph: generateNoice(batch_size, 100),
                            image_ph: image_handler.getBatchImage()
                        }
                        _discriminator_loss, _ = sess.run([net.discriminator_loss, net.discriminator_optimize], feed_dict=feed_dict)
                        _generator_loss, _ = sess.run([net.generator_loss, net.generator_optimize], feed_dict=feed_dict)
                        _discriminator_loss_list.append(_discriminator_loss)
                    _generator_loss, _ = sess.run([net.generator_loss, net.generator_optimize], feed_dict=feed_dict)
                    if _generator_loss == float('nan'):
                        break
                    _generator_loss_list.append(_generator_loss)
                    if j > image_handler.getImageNum() // batch_size // 10:
                        break
                
                # Record and show
                generator_loss = np.mean(np.asarray(_discriminator_loss_list))
                discriminator_loss = np.mean(np.asarray(_generator_loss_list))
                generator_loss_list.append(generator_loss)
                discriminator_loss_list.append(discriminator_loss)
                print('iter: ', i, '\tgenerator loss: ', generator_loss, '\tdiscriminator loss: ', discriminator_loss)

                # Store Result
                columns = ['epoch', 'g_loss', 'd_loss']
                df = pd.DataFrame(0, index = range(i+1) ,columns = columns)
                df['epoch'] = range(i+1)
                df['g_loss'] = generator_loss_list
                df['d_loss'] = discriminator_loss_list
                df.to_csv(output_csv_dir)

                # Visualize
                if i % 20 == 0:
                    feed_dict = {
                        noise_ph: generateNoice(batch_size, 100),
                    }
                    generated_image = sess.run([net.generated_tensor], feed_dict=feed_dict)
                    generated_image = generated_image[0] * 255
                    saveGeneratedBatch(generated_image, 8, i, output_img_dir)