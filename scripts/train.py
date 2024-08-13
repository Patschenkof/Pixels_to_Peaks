import sys
import datetime
import tensorflow as tf
import tensorflow.keras as keras
from keras.optimizers.schedules import ExponentialDecay
sys.path.append('..')
from scripts.inpaint_ops import d_wasserstein_loss, g_wasserstein_loss, min_max_tf_scaler2, mask_2_crop
from scripts.data_processing_utils import DataLoader
from models.networks.generator import dem_fill_net
from models.networks.global_critic import build_wgan_discriminator_global
from models.networks.local_critic import build_wgan_discriminator_local
from models.inpaintModel import WGAN_GP, GANMonitor
import yaml
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Parser for path of config file')

# Define the arguments
parser.add_argument('--config', type = str, help = 'Path to the config file')

# Parse the arguments
args = parser.parse_args()
config_path = args.config

# Initialize Data Loader
DataLoader = DataLoader()



with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Define Model Parameters
batch_size = config['batch_size']
log_dir = config['log_dir']
save_gen_path = config['save_gen_path']
NUM_EPOCHS = config['num_epochs']
LR = config['learning_rate']
l1_alpha = config['l1_alpha']
global_wgan_loss_alpha = config['global_wgan_loss_alpha']
gp_weight = config['gp_weight']
gan_loss_alpha = config['gan_loss_alpha']
l1_loss_alpha = config['l1_loss_alpha']
ae_loss_alpha = config['ae_loss_alpha']


large_dim = config['large_dim']
small_dim = config['small_dim']


# Load the data

dem_path = config['dem_path']
inner_outer_path = config['inner_outer_path']
intersection_path = config['intersection_path']
intersection_small_path = config['intersection_small_path']

# Define the lists:

dem_list = DataLoader.populate_list(dem_path)
inner_outer_list = DataLoader.populate_list(inner_outer_path)
intersection_list = DataLoader.populate_list(intersection_path)
intersection_small_list = DataLoader.populate_list(intersection_small_path)

# shuffle the lists
dem_list, inner_outer_list, intersection_list, intersection_small_list = DataLoader.shuffle_lists([dem_list, inner_outer_list, intersection_list,
                                                                                                   intersection_small_list])

# Load the data
dem_tensor = tf.data.Dataset.from_generator(
    DataLoader.load_dem_cv2,
    args=[dem_list],
    output_signature = tf.TensorSpec(shape = (large_dim,large_dim,1), dtype = tf.float32)
).batch(batch_size)

train_inner_outer = tf.data.Dataset.from_generator(
    DataLoader.load_mask,
    args=[inner_outer_list],
    output_signature = tf.TensorSpec(shape = (large_dim,large_dim,1), dtype = tf.float32)
).batch(batch_size)

train_intersection = tf.data.Dataset.from_generator(
    DataLoader.load_mask,
    args=[intersection_list],
    output_signature = tf.TensorSpec(shape = (large_dim,large_dim,1), dtype = tf.float32)
).batch(batch_size)

train_intersection_small = tf.data.Dataset.from_generator(
    DataLoader.load_mask,
    args=[intersection_small_list, (small_dim,small_dim)],
    output_signature = tf.TensorSpec(shape = (small_dim,small_dim,1), dtype = tf.float32)
).batch(batch_size)

# Normalize the DEMs

train_norm_dems = dem_tensor.map(min_max_tf_scaler2, num_parallel_calls = tf.data.AUTOTUNE)
#train_norm_dems = dem_tensor

# Define the model:

generator = dem_fill_net(image_size=large_dim)
global_discriminator = build_wgan_discriminator_global(image_size=large_dim)
local_discriminator = build_wgan_discriminator_local(image_size=small_dim)

wgan = WGAN_GP(generator = generator, global_critic = global_discriminator, local_critic = local_discriminator, l1_alpha=l1_alpha,
                global_wgan_loss_alpha = global_wgan_loss_alpha, gp_weight = gp_weight, gan_loss_alpha = gan_loss_alpha,
                l1_loss_alpha = l1_loss_alpha, ae_loss_alpha = ae_loss_alpha)

# set the prefetch buffer size
train_dem = train_norm_dems.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
train_inner_outer = train_inner_outer.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
train_intersection = train_intersection.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
train_intersection_small = train_intersection_small.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

# zip the datasets

dataset = tf.data.Dataset.zip((train_dem, train_inner_outer, train_intersection,
                               train_intersection_small))

# Define the callbacks:



gan_monitor = GANMonitor(log_dir = log_dir,
                        #save_img_epoch = save_img_epoch,
                        save_gen_path = save_gen_path,
                        data = dataset)


LR_sheduler = ExponentialDecay(LR, decay_steps = 100000, decay_rate = 0.96, staircase = True)

# define the optimizers
d_optimzer_local = keras.optimizers.Adam(learning_rate = LR_sheduler, beta_1 = 0.5, beta_2 = 0.999)
d_optimizer_global = keras.optimizers.Adam(learning_rate = LR_sheduler, beta_1 = 0.5, beta_2 = 0.999)
g_optimizer = keras.optimizers.Adam(learning_rate = LR_sheduler, beta_1 = 0.5, beta_2 = 0.999)

# define the loss functions
d_loss_fn = d_wasserstein_loss
g_loss_fn = g_wasserstein_loss

# compile the models

wgan.compile(d_optimizer_local = d_optimzer_local, d_optimizer_global = d_optimizer_global, g_optimizer = g_optimizer,
                d_loss_fn = d_loss_fn, g_loss_fn = g_loss_fn)


# Train the model

wgan.fit(dataset, epochs = NUM_EPOCHS, callbacks = [gan_monitor])
