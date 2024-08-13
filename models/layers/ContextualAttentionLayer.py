import tensorflow as tf
from tensorflow.keras.layers import Layer
#import tensorflow.keras.backend as K
import tensorflow.keras as keras
import sys
sys.path.append('..')
from scripts.inpaint_ops import batched_conv

######################################################################################


class ContextualAttentionLayer(Layer):
  # This layer is the implementation of the Contextual Attention Layer as described in the paper
  # "Contextual Attention for Image Inpainting" by Yu et al.
  # https://github.com/JiahuiYu/generative_inpainting  
  
    def __init__(self, ksize = 3, stride = 1, rate = 2, fuse_k = 3, softmax_scale = 10., **kwargs):
        super(ContextualAttentionLayer, self).__init__(**kwargs)
        self.ksize = ksize
        self.stride = stride
        self.fuse_k = fuse_k
        self.rate = rate
        self.softmax_scale = softmax_scale

    def build(self, input_shape):
      #

      # Create a trainable weight variable for this layer.
      self.fuse_weight = self.add_weight(name='fuse_weight',
                                            shape=(self.fuse_k, self.fuse_k, 1, 1),
                                            initializer='glorot_uniform',
                                            trainable=True)
      #super(ContextualAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, f,b,mask):

      # get shapes
      raw_int_fs = tf.cast(tf.shape(f), tf.int32)
      raw_int_bs = tf.cast(tf.shape(b), tf.int32)
      raw_int_mask = tf.cast(tf.shape(mask), tf.int32)
      raw_fs = tf.shape(f)
      kernel = 2 * self.rate


      ########################################### Extract Patches from background ################################
      raw_w = tf.image.extract_patches(b, [1,kernel, kernel,1],[1,self.rate*self.stride, self.rate*self.stride,1], [1,1,1,1], padding = 'SAME')
      raw_w = tf.reshape(raw_w, [raw_int_bs[0], tf.shape(raw_w)[1]*tf.shape(raw_w)[2], kernel, kernel, raw_int_bs[-1]]) # [None, 1024,4,4,64]
      raw_w = tf.transpose(a=raw_w, perm = [0,2,3,4,1]) # b,k,k,c,hw -- hw denotes number of patches [None, 4,4,64,1024]
      ################################################################


      ######################tf.resize####################################WORKS
      """
      They resize using scale = 1. / self.rate which results in 1/2 in their code. This leads to a floating error in tensorflow, so I'll just half the image dim
      another way
      """
      f = tf.image.resize(f, [raw_int_fs[1] // 2, raw_int_fs[2] // 2], method = 'nearest')
      b = tf.image.resize(b, [raw_int_bs[1] // 2, raw_int_bs[2] // 2], method = 'nearest')
      mask = tf.image.resize(mask, [raw_int_mask[1] //2, raw_int_mask[2] // 2], method = 'nearest')
      ###################################################################

      ############################tf.resize###################################
      """
      Get new shapes
      """
      int_fs = tf.cast(tf.shape(f), tf.int32)
      int_bs = tf.cast(tf.shape(b), tf.int32)
      int_mask_s = tf.cast(tf.shape(mask), tf.int32)

      fs = tf.shape(f)
      bs = tf.shape(b)
      mask_s = tf.shape(mask)
      ########################################################################

      ############################################ Extract Patches from resized Background ############################
      """
      Same as with raw_w, only that b is now smaller (32,32,64)
      """
      w = tf.image.extract_patches(b, [1,self.ksize,self.ksize,1], [1, self.stride, self.stride, 1], [1,1,1,1], padding = 'SAME')# [None, 32,32,576]
      w = tf.reshape(w, [int_fs[0],tf.shape(w)[1]*tf.shape(w)[2] , self.ksize, self.ksize, int_fs[-1]])# [None, 1024,3,3,64]
      w = tf.transpose(a=w, perm = [0,2,3,4,1]) # [None, 3,3,64,1024] again [Batch_size, kernel, kernel, channels, number of patches]
      ###############################################################################

      ############################################ Extract Patches from resized mask ##############################
      """
      Same as with w and raw_w
      """
      m = tf.image.extract_patches(mask, [1,self.ksize, self.ksize, 1], [1,self.stride,self.stride,1], [1,1,1,1], padding = 'SAME')#[None, 32,32,576]
      m = tf.reshape(m, [int_mask_s[0],tf.shape(m)[1] * tf.shape(m)[2],self.ksize,self.ksize, int_mask_s[-1]])# [None, 1024,3,3,64]
      m = tf.transpose(a=m, perm = [0,2,3,4,1])# [None, 3,3,64,1024]
      #######################################################################

      mm = tf.cast(tf.equal(tf.reduce_mean(input_tensor=m, axis=[1,2,3], keepdims=True), 0.), tf.float32)
      scale = self.softmax_scale

      yi = tf.vectorized_map(lambda x: batched_conv(x, fs, bs, raw_fs, scale, self.fuse_weight, self.rate), (f, w, raw_w, mm))
      yi = tf.squeeze(yi, axis = 1)

      return yi