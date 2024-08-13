# Abstrahiert von dem-fill inpaint_ops

import cv2
import numpy as np
import tensorflow as tf
import os


#################################################################################
# Function for tf.vectorized_map:

def batched_conv(inputs, fs, bs, raw_fs, scale, fuse_weight, rate):
  """
  This function is part of the Contextual Attention Layer. This was first introduced in the paper:
    "Generative Image Inpainting With Contextual Attention" by Yu et al.
    https://github.com/JiahuiYu/generative_inpainting


  Args:
    inputs: tuple of (f, w, raw_w, mm), foreground, background patches, raw background patches, meaned mask
    fs: shape of f
    bs: shape of b
    raw_fs: shape of raw_f
    scale: softmax scale
    fuse_weight: weight for fusion
    rate: rate for resize
  """
  f, w, raw_w, mm = inputs

  # Expand f with one dummy dim:
  f = tf.expand_dims(f, axis = 0) # [1,32,32,64]

  # Normalize wi
  wi = w
  wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(input_tensor=tf.square(wi), axis=[0,1,2])), 1e-4)

  yi = tf.nn.conv2d(f, wi_normed, strides = [1,1,1,1], padding = 'SAME')
  yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
  yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
  yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
  yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
  yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
  yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
  yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
  yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
  yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

  yi *= mm
  yi = tf.nn.softmax(yi*scale, 3)
  yi *= mm 
  yi = tf.nn.conv2d_transpose(yi, raw_w, tf.concat([[1], raw_fs[1:]], axis = 0), strides=[1,rate,rate,1]) / 4.

  return yi
######################################################################################

####################################################################################
# function to get image for the local discriminator

def mask_2_cropLEGACY(image, mask_inner_outer, intersection_mask_small):

    """
    Function to crop the image based on the mask. The cropped image can then be fed into the local discriminator.

    Parameters:
    image : tf.Tensor
        Tensor containing the image, which should be cropped

    mask_inner_outer : tf.Tensor
        Tensor containing the mask, which should be used to crop the image

    intersection_mask_small : tf.Tensor
        Just needed to get the shape of the resulting image
    """

    # Get the indices of the mask
    indices = tf.where(mask_inner_outer == 1) # Glacier is initialized as 1. So we can (1. - mask) to extract the glacier

    # Get shape of the resultin mask: (64, 64)
    shapex = tf.get_static_value(tf.shape(intersection_mask_small)[0])
    shapey = tf.get_static_value(tf.shape(intersection_mask_small)[1])

    # Get the top left indices of our mask, which can be used by tensorflow to crop the image
    top_left = tf.reduce_min(indices, axis = 0)
    x1, y1 = tf.get_static_value(top_left[0]), tf.get_static_value(top_left[1])

    # Crop the image
    cropped = tf.image.crop_to_bounding_box(image, x1, y1, shapex, shapey)

    return cropped

####################################################################################

####################################################################################
# function to get image for the local discriminator

@tf.function
def mask_2_crop(inputs):
  

  image, mask_inner_outer, intersection_mask_small = inputs

  indices = tf.where(mask_inner_outer == 1)

  shapex = tf.shape(intersection_mask_small)[0]
  shapey = tf.shape(intersection_mask_small)[1]

  shapex = tf.cast(shapex, dtype = tf.int32)
  shapey = tf.cast(shapey, dtype = tf.int32)
  
  top_left = tf.reduce_min(indices, axis = 0)

  x1 = tf.cast(top_left[0], tf.int32)
  y1 = tf.cast(top_left[1], tf.int32)

  cropped = tf.image.crop_to_bounding_box(image, x1, y1, shapex, shapey)

  return cropped


#################################################################
# Custom Loss function for WGAN
# From: https://github.com/margaretmz/GANs-in-Art-and-Design/blob/main/4_wgan_gp_anime_faces.ipynb
# Wasserstein loss for the critic
@tf.function
def d_wasserstein_loss(pred_real, pred_fake):

    return tf.reduce_mean(pred_fake - pred_real)

#################################################################
# Wasserstein loss for the generator
# Adapted from:
# Neuralgym: https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/gan_ops.py
@tf.function
def g_wasserstein_loss(pred_fake):
    return -tf.reduce_mean(pred_fake)


############################################################################
def min_max_tf_scaler2(x):
    '''
    Function to scale the input tensor to the range [-1,1]
    '''
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    
    return ((x - min_val) / (max_val - min_val)) * 2.  -1.

def min_max_tf_scaler(x):
    '''
    Function to scale the input tensor to the range [0,1]
    
    '''
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    return (x - min_val) / (max_val - min_val)

############################################################################
# Spatial Discounting Mask:

@tf.function
def spatial_discounting_mask(mask, gamma = 0.9):
    gamma = tf.cast(gamma, tf.float32)
    shape = tf.shape(mask)    

    # Get coordinates of masked pixels
    masked_indices = tf.where(mask == 1.0)

    center_x = tf.math.reduce_mean(masked_indices[:, 0]) #(i)
    center_y = tf.math.reduce_mean(masked_indices[:, 1]) #(j)
    mask_values = tf.ones(shape, dtype = tf.float32)

    # Create meshgrid (Good for vectorized operations)
    #i,j = tf.cast(tf.meshgrid(tf.range(shape[0]), tf.range(shape[1])), tf.float32)
    i, j = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing = 'ij')
    i = tf.cast(i, tf.float32)
    j = tf.cast(j, tf.float32)
    dist_x = tf.cast(tf.abs(i - tf.cast(center_x, tf.float32)), tf.float32)
    dist_y = tf.cast(tf.abs(j - tf.cast(center_y, tf.float32)), tf.float32)

    # Calculate the euclidean distance
    c_squared = tf.sqrt(tf.math.pow(dist_x, 2) + tf.math.pow(dist_y, 2))

    # Calculate the discounting mask
    mask_values = tf.cast(tf.pow(gamma, gamma - c_squared), tf.float32) # Gamma - c_squared to invert the discounting mask

    #mask_constant = tf.constant(mask_values, shape = shape, dtype = tf.float32)

    return mask_values


@tf.function
def spatial_discounting_mask_orig(mask, gamma = 0.9):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    shape = tf.shape(mask)
    mask_values = tf.ones(shape, tf.float32)
    gamma = tf.cast(gamma, tf.float32)

    # Create meshgrid (Good for vectorized operations)
    center_x = tf.cast(shape[0] , tf.float32)
    center_y = tf.cast(shape[1] , tf.float32)
    i, j = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing = 'ij')
    i = tf.cast(i, tf.float32)
    j = tf.cast(j, tf.float32)

    mask_values = tf.math.maximum(tf.pow(gamma, tf.math.minimum(i, center_x-i)), tf.pow(gamma, tf.math.minimum(j, center_y-j)))
    

    return tf.cast(mask_values, dtype=tf.float32)
