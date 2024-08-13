import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import tensorflow.keras as keras
from models.layers.ContextualAttentionLayer import ContextualAttentionLayer

def dem_fill_net(filters = 16, image_size = 256):

  """
  Generator for the dem fill network. This network is used to fill in the missing values in the DEM.

  The Network is based on the following paper:
  Void Filling of Digital Elevation Models with Deep Generative Models, Gavrtiil et al. 2019

  """

  dem = keras.Input(shape = (image_size,image_size,1), name = 'dem')
  inner_outer = keras.Input(shape = (image_size,image_size,1), name = 'inner_outer')
  intersection = keras.Input(shape = (image_size,image_size,1), name = 'intersection')

  masked_dem = tf.math.multiply(dem,(1. - intersection))
  xin = masked_dem
  ones_x = tf.ones_like(dem)
  input = tf.concat([masked_dem, ones_x, intersection], axis = -1)

  # stage 1 network
  x = layers.Conv2D(filters, (5,5), strides = (1,1), padding = 'same',
                    name = 'conv1', activation = tf.nn.elu)(input)


  # Downsample input (256,256) --> (128,128)
  x = layers.Conv2D(filters * 2, (3,3), strides = (2,2), padding = 'same',
                      name = 'conv2_down', activation = tf.nn.elu)(x)

  # Keeps dims:
  x = layers.Conv2D(filters * 2, (3,3), strides = (1,1), padding = 'same',
                      name = 'conv3', activation = tf.nn.elu)(x)

  # Downsample input (128,128) --> (64,64)
  x = layers.Conv2D(filters * 4, (3,3), strides = (2,2), padding = 'same',
                      name = 'conv4_down', activation = tf.nn.elu)(x)

  # Keeps dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                      name = 'conv5', activation = tf.nn.elu)(x)

  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                      name = 'conv6', activation = tf.nn.elu)(x)

  # Resize inner_outer to dim of x

  mask_s = tf.image.resize(inner_outer, size = x.get_shape().as_list()[1:3], method = 'nearest')

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (2,2), name = 'conv7_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (4,4), name = 'conv8_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (8,8), name = 'conv9_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (8,8), name = 'conv10_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (4,4), name = 'conv11_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (2,2), name = 'conv12_dilated',
                    activation = tf.nn.elu)(x)

  # Followed by normal and upsampling convolutions:

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'conv13', activation = tf.nn.elu)(x)

  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'conv14', activation = tf.nn.elu)(x)

  # Upsample input (64,64) --> (128,128)
  x = layers.Conv2DTranspose(filters * 2, (3,3), strides = (2,2), padding = 'same',
                              name = 'conv15_up', activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 2, (3,3), strides = (1,1), padding = 'same',
                    name = 'conv16', activation = tf.nn.elu)(x)

  # Upsample input (128,128) --> (256,256)
  x = layers.Conv2DTranspose(filters, (3,3), strides = (2,2), padding = 'same',
                              name = 'conv17_up', activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters, (3,3), strides = (1,1), padding = 'same',
                    name = 'conv18', activation = tf.nn.elu)(x)

  # Output layer:
  x = layers.Conv2D(1, (3,3), strides = (1,1), padding = 'same',
                    name = 'conv19', activation = None)(x)


  x = tf.clip_by_value(x, -1., 1.)
  x_stage1 = x

  # Stage 2 network.

  x = tf.math.multiply(x, intersection) + tf.math.multiply(xin, (1. - intersection))

  xnow = tf.concat([x,  ones_x, intersection], axis=-1)

  # Convolution branch:

  # Keeps dims:
  x = layers.Conv2D(filters, (5,5), strides = (1,1), padding = 'same',
                    name = 'xconv1', activation = tf.nn.elu)(xnow) # xnow

  # Downsample input (256,256) --> (128,128)
  x = layers.Conv2D(filters * 2, (3,3), strides = (2,2), padding = 'same',
                      name = 'xconv2_down', activation = tf.nn.elu)(x)

  # Keeps dims:
  x = layers.Conv2D(filters * 2, (3,3), strides = (1,1), padding = 'same',
                      name = 'xconv3', activation = tf.nn.elu)(x)

  # Downsample input (128,128) --> (64,64)
  x = layers.Conv2D(filters * 4, (3,3), strides = (2,2), padding = 'same',
                      name = 'xconv4_down', activation = tf.nn.elu)(x)

  # Keeps dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                      name = 'xconv5', activation = tf.nn.elu)(x)

  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                      name = 'xconv6', activation = tf.nn.elu)(x)

  # Dialated convolutions:

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (2,2), name = 'xconv7_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (4,4), name = 'xconv8_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (8,8), name = 'xconv9_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (8,8), name = 'xconv10_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (4,4), name = 'xconv11_dilated',
                    activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    dilation_rate = (2,2), name = 'xconv12_dilated',
                    activation = tf.nn.elu)(x)

  # Normal convolutions:
  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'xconv13', activation = tf.nn.elu)(x)

  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'xconv14', activation = tf.nn.elu)(x)
  x_hallu = x

  # Attention branch:
  # Keeps Dims:
  x = layers.Conv2D(filters, (5,5), strides = (1,1), padding = 'same',
                    name = 'pmconv1', activation = tf.nn.elu)(xnow)

  # Downsample input (256,256) --> (128,128)
  x = layers.Conv2D(filters * 2, (3,3), strides = (2,2), padding = 'same',
                      name = 'pmconv2_down', activation = tf.nn.elu)(x)

  # Keeps dims:
  x = layers.Conv2D(filters * 2, (3,3), strides = (1,1), padding = 'same',
                      name = 'pmconv3', activation = tf.nn.elu)(x)

  # Downsample input (128,128) --> (64,64)
  x = layers.Conv2D(filters * 4, (3,3), strides = (2,2), padding = 'same',
                      name = 'pmconv4_down', activation = tf.nn.elu)(x)

  # Keeps dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                      name = 'pmconv5', activation = tf.nn.elu)(x)

  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                      name = 'pmconv6', activation = tf.nn.relu)(x)

  # Contextual attention mechanism
  x = ContextualAttentionLayer()(x,x,mask_s)

  # Keeps dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'pmconv11', activation = tf.nn.elu)(x)

  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'pmconv12', activation = tf.nn.elu)(x)

  # Prediction mask maybe?!
  pm = x
  x = tf.concat([x_hallu, pm], axis = -1)

  # Keeps dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'allconv13', activation = tf.nn.elu)(x)

  # keep dims:
  x = layers.Conv2D(filters * 4, (3,3), strides = (1,1), padding = 'same',
                    name = 'allconv14', activation = tf.nn.elu)(x)

  # Upsample input (64,64) --> (128,128)
  x = layers.Conv2DTranspose(filters * 2, (3,3), strides = (2,2), padding = 'same',
                              name = 'allconv15_up', activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters * 2, (3,3), strides = (1,1), padding = 'same',
                    name = 'allconv16', activation = tf.nn.elu)(x)

  # Upsample input (128,128) --> (256,256)
  x = layers.Conv2DTranspose(filters, (3,3), strides = (2,2), padding = 'same',
                              name = 'allconv17_up', activation = tf.nn.elu)(x)

  # Keep dims:
  x = layers.Conv2D(filters // 2, (3,3), strides = (1,1), padding = 'same',
                    name = 'allconv18', activation = tf.nn.elu)(x)

  # Output layer:
  x = layers.Conv2D(1, (3,3), strides = (1,1), padding = 'same',
                    name = 'allconv19', activation = None)(x)

  x_stage2 = tf.clip_by_value(x, -1., 1.)


  model = keras.Model(inputs = [dem, inner_outer, intersection], outputs = [x_stage1, x_stage2])

  return model
