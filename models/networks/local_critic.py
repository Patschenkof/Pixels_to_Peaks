import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def build_wgan_discriminator_local(filters = 32, image_size = 64):
  """
  Discriminator for the WGAN-GP model
  Used to classify the input image as real or fake. Introduced in the WGAN paper:
  K. Gavriil, O.J.D. Barrowclough, G. Muntingh, Void Filling of Digital Elevation Models with Deep Generative Models
  """

  dem = keras.Input(shape = (image_size,image_size,1), name = 'dem')
  intersection_small = keras.Input(shape = (image_size,image_size,1), name = 'intersection_small')

  #input = tf.concat([dem, intersection_small], axis = -1)
  input = dem


  # (64,64) --> (32,32)
  x = layers.Conv2D(filters, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv1')(input)

  # (32,32) --> (16,16)
  x = layers.Conv2D(filters * 2, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv2')(x)

  # (16,16) --> (8,8)
  x = layers.Conv2D(filters * 4, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv3')(x)

  # (8,8) --> (4,4)
  x = layers.Conv2D(filters * 8, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv4')(x)

  # Flatten the input
  x = layers.Flatten()(x)

  #Dense layer
  x = layers.Dense(1, name = 'local_dense')(x)

  x_out = x

  model = keras.Model(inputs = [dem, intersection_small], outputs = x_out)

  return model