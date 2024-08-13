import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import sys
sys.path.append('..')
import os   
from scripts.inpaint_ops import mask_2_crop, spatial_discounting_mask_orig
import matplotlib.pyplot as plt
     

##################################################################
# Custom WGAN_GP Model, adapted from:
# https://github.com/margaretmz/GANs-in-Art-and-Design/blob/main/4_wgan_gp_anime_faces.ipynb

# The model is a combination of a global critic, a local critic and a generator. Code logic has been adapted from 
# Yu et al. (2018) "Generative Image Inpainting with Contextual Attention" and from Gavriil et al. (2016)
# "Void Filling of Digital Elevation Models with Deep Generative Models"

class WGAN_GP(keras.Model):
  def __init__(self, local_critic, global_critic, generator, l1_alpha = 1.2, global_wgan_loss_alpha = 1., gp_weight = 10.0, gan_loss_alpha = 0.001, l1_loss_alpha = 1.2, ae_loss_alpha = 1.2):
    super().__init__()
    self.local_critic = local_critic
    self.global_critic = global_critic
    self.generator = generator
    self.gp_weight = gp_weight 
    self.l1_alpha = l1_alpha
    self.global_wgan_loss_alpha = global_wgan_loss_alpha
    self.gan_loss_alpha = gan_loss_alpha
    self.l1_loss_alpha = l1_loss_alpha
    self.ae_loss_alpha = ae_loss_alpha

    # Metrics:
    self.d_global_loss_metric = keras.metrics.Mean(name='d_global_loss')
    self.d_local_loss_metric = keras.metrics.Mean(name='d_local_loss')
    self.g_loss_metric = keras.metrics.Mean(name='g_loss')

  def compile(self, d_optimizer_local, d_optimizer_global, g_optimizer, d_loss_fn, g_loss_fn):
    super(WGAN_GP, self).compile()
    self.d_optimizer_local = d_optimizer_local
    self.d_optimizer_global = d_optimizer_global
    self.g_optimizer = g_optimizer
    self.d_loss_fn = d_loss_fn
    self.g_loss_fn = g_loss_fn


  def random_interpolates(self, batch_size, real_images, fake_images):
     # Random Interpolates code. Adapted from Neuralgym ops.gan_ops
     # https://github.com/JiahuiYu/neuralgym/tree/master
     
     shape = tf.shape(real_images)

     x = tf.reshape(real_images, [batch_size, -1])
     y = tf.reshape(fake_images, [batch_size, -1])

     alpha = tf.random.uniform(shape=[batch_size, 1])

     interpolates = x + alpha * (y - x)

     return tf.reshape(interpolates, shape)
  
  def gradients_penalty(self, interpolates, critic_fn, mask, norm = 1.):   
    # Gradient Penalty code. Adapted from Neuralgym ops.gan_ops
    # https://github.com/JiahuiYu/neuralgym/tree/master 

    with tf.GradientTape() as tape:
      tape.watch(interpolates)
      pred = critic_fn([interpolates, mask]) 

    gradients = tape.gradient(pred, [interpolates])[0]

    
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis = [1,2,3]))
    return tf.reduce_mean(tf.square(slopes - norm))

  @property
  def metrics(self):
    return [self.d_global_loss_metric, self.d_local_loss_metric, self.g_loss_metric]

  @tf.function
  def train_step(self,data):

    batch_pos, inner_outer, intersection, intersection_small = data
    local_patch_batch_pos = tf.vectorized_map(mask_2_crop, (batch_pos, inner_outer, intersection_small)) 
    batch_size = tf.shape(batch_pos)[0]

    # Resize the inner_outer mask to the size of the intersection_small mask
    # Used for spatial discounting mask
    mask_s = tf.image.resize(inner_outer, size = intersection_small.get_shape().as_list()[1:3], method = 'nearest')


    # train the global critic:
    with tf.GradientTape() as tape_global, tf.GradientTape() as tape_local, tf.GradientTape() as tape_gen:
      # let the geneartor generate an output (usable for both local and global critic)
      output_stage1, output_stage2 = self.generator([batch_pos, inner_outer, intersection], training = True) 
      batch_predicted = output_stage2

      # Create a combination of generated glacier and original dem
      batch_complete = tf.stop_gradient(tf.math.multiply(batch_predicted, intersection) + tf.math.multiply(batch_pos, (1. - intersection)))

      ################################ Patch creation Logic ##################################################################################

      # Get a local patch from the batch_complete batch:
      local_patch_batch_complete = tf.stop_gradient(tf.vectorized_map(mask_2_crop, (batch_complete, inner_outer, intersection_small)))

      # Get a local patch from batch_predicted:
      local_patch_batch_predicted = tf.stop_gradient(tf.vectorized_map(mask_2_crop, (batch_predicted, inner_outer, intersection_small)))
      local_patch_x2 = local_patch_batch_predicted

      # Get local patch from stage 1 Network:
      local_patch_x1 = tf.stop_gradient(tf.vectorized_map(mask_2_crop, (output_stage1, inner_outer, intersection_small)))
      #######################################################################################################################################

      ################################# Let Discriminators make a guess ################################################################
      # let the global critic make a prediction:
      pos_global = self.global_critic([batch_pos, intersection], training = True)
      neg_global = self.global_critic([batch_complete, intersection], training = True)

      # let the local critic make a prediction:
      pos_local = self.local_critic([local_patch_batch_pos, intersection_small], training = True)
      neg_local = self.local_critic([local_patch_batch_complete, intersection_small], training = True)

      interpolates_local = self.random_interpolates(batch_size, local_patch_batch_pos, local_patch_batch_complete)
      interpolates_global = self.random_interpolates(batch_size, batch_pos, batch_complete)

      gp_local = self.gradients_penalty(interpolates_local, self.local_critic, mask_s)
      gp_global = self.gradients_penalty(interpolates_global, self.global_critic, inner_outer)

      # Add gradient penalty to the critics loss:
      d_loss_global = self.d_loss_fn(pos_global, neg_global) + gp_global * self.gp_weight

      # Add gradient penalty to the critics loss
      d_loss_local = self.d_loss_fn(pos_local, neg_local) + gp_local * self.gp_weight # Same as Above!

      ###################################################################################################################################

      ############################## Evaluate Generator Performance ####################################################################

      pred_fake_local = neg_local
      pred_fake_global = neg_global

      g_loss_local, g_loss_global = self.g_loss_fn(pred_fake_local), self.g_loss_fn(pred_fake_global)

      g_loss = self.global_wgan_loss_alpha * g_loss_global + g_loss_local

      g_loss = g_loss * self.gan_loss_alpha

      #L1-loss:
      l1_loss = self.l1_alpha * tf.reduce_mean(input_tensor=tf.abs(local_patch_batch_pos - local_patch_x1)) * tf.vectorized_map(spatial_discounting_mask_orig, mask_s) 
      l1_loss += tf.reduce_mean(input_tensor=tf.abs(local_patch_batch_pos - local_patch_x2)) * tf.vectorized_map(spatial_discounting_mask_orig, mask_s) 

      # ae_loss:
      ae_loss = self.l1_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_pos - output_stage1) * (1.- intersection))
      ae_loss += tf.reduce_mean(input_tensor=tf.abs(batch_pos - output_stage2) * (1. - intersection))
      ae_loss /= tf.reduce_mean(input_tensor=1.- intersection)

      # Add both losses
      g_loss += self.l1_loss_alpha * l1_loss
      g_loss += self.ae_loss_alpha * ae_loss


    # Compute gradients for global critic:
    grads_global = tape_global.gradient(d_loss_global, self.global_critic.trainable_variables)
    self.d_optimizer_global.apply_gradients(zip(grads_global, self.global_critic.trainable_variables))

    # Compute gradients for local critic:
    grads_local = tape_local.gradient(d_loss_local, self.local_critic.trainable_variables)
    self.d_optimizer_local.apply_gradients(zip(grads_local, self.local_critic.trainable_variables))

    # Compute gradients for generator:
    grads_gen = tape_gen.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))

    self.d_local_loss_metric.update_state(d_loss_local)
    self.d_global_loss_metric.update_state(d_loss_global)
    self.g_loss_metric.update_state(g_loss)

    return {'d_local_loss': self.d_local_loss_metric.result(), 'd_global_loss': self.d_global_loss_metric.result(), 'g_loss': self.g_loss_metric.result()}
    
  




class GANMonitor(tf.keras.callbacks.Callback): 
  # Custom Callback to monitor the training of the GAN
  # Adapted from: https://github.com/margaretmz/GANs-in-Art-and-Design/blob/main/4_wgan_gp_anime_faces.ipynb
  
  def __init__(self, log_dir = None, save_gen_path = None, data = None):
      super().__init__()
      self.log_dir = log_dir
      self.save_gen_path = save_gen_path
      self.data = data
      self.summary_writer = tf.summary.create_file_writer(self.log_dir)

      if save_gen_path is None:
        raise ValueError('No path to save the generator has been specified. Please specify path')

      if log_dir is None:
        raise ValueError('No path for the logs has been specified')
      
        

  def on_epoch_end(self, epoch, logs=None):
      batch_pos, inner_outer, intersection, intersection_small = next(iter(self.data))
      x1, x2 = self.model.generator([batch_pos, inner_outer, intersection])
      cropped = tf.vectorized_map(mask_2_crop, (x2, inner_outer, intersection_small))

    
      # Save scalar values for Tensorboard:        
      with self.summary_writer.as_default():

        tf.summary.scalar('d_local_loss', self.model.d_local_loss_metric.result(), step = epoch)
        tf.summary.scalar('d_global_loss', self.model.d_global_loss_metric.result(), step = epoch)
        tf.summary.scalar('g_loss', self.model.g_loss_metric.result(), step = epoch)
        tf.summary.scalar('LR', self.model.g_optimizer.lr, step = epoch)
        tf.summary.image('batch_pos (Original DEM)', batch_pos, step = epoch)
        tf.summary.image('intersection', intersection, step = epoch)
        tf.summary.image('Masked DEM', tf.math.multiply(batch_pos, (1. - intersection)), step = epoch)
        tf.summary.image('Stage 2 output', x2, step = epoch)
        tf.summary.image('cropped for local critic', cropped, step = epoch)
        tf.summary.image('Stage 2 + prod. Batch', tf.math.multiply(x2, intersection) + tf.math.multiply(batch_pos, (1. - intersection)), step = epoch)


      # Save the generator weights:
      self.model.generator.save_weights(os.path.join(self.save_gen_path, f'generator_{epoch}.keras'.format(epoch)))


  def on_train_end(self, logs=None):
      self.model.generator.save_weights(os.path.join(self.save_gen_path, f'generator_final.keras'))
      #self.model.generator.save(os.path.join(self.save_gen_path, 'generator.keras'))