import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def build_wgan_discriminator_global(filters = 32, image_size = 256):
  """
  Discriminator for the WGAN-GP model
  Used to classify the input image as real or fake. Introduced in the WGAN paper:
  K. Gavriil, O.J.D. Barrowclough, G. Muntingh, "Void Filling of Digital Elevation Models with Deep Generative Models"
  """

  dem = keras.Input(shape = (image_size,image_size,1))
  intersection = keras.Input(shape = (image_size,image_size,1))

  #input = tf.concat([dem, intersection], axis = -1)
  input = dem

  # Downsample input (256,256) --> (128,128)
  x = layers.Conv2D(filters, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv1')(input)
  #x = tf.nn.leaky_relu(x)

  # Downsample input (128,128) --> (64,64)
  x = layers.Conv2D(filters * 2, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv2')(x)
  #x = tf.nn.leaky_relu(x)

  # Downsample input (64,64) --> (32,32)
  x = layers.Conv2D(filters * 4, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv3')(x)
  #x = tf.nn.leaky_relu(x)

  # Downsample input (32,32) --> (16,16)
  x = layers.Conv2D(filters * 4, (5,5), strides = (2,2), padding = 'same', name = 'disc_conv4')(x)
  #x = tf.nn.leaky_relu(x)

  # Flatten the output:
  x = layers.Flatten()(x)

  # Dense layer (in dem-fill at build_wgan_discriminator)
  x = layers.Dense(1, name = 'global_dense')(x)

  x_out = x

  model = keras.Model(inputs = [dem, intersection], outputs = x_out)
  return model