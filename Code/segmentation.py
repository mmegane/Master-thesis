seed = 0

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.compat.v1.set_random_seed(seed)

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

import warnings

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category = FutureWarning)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

# Allow growth of GPU memory (otherwise it will look like all the memory is being used, even if you only use 10 MB)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

#%%

LOAD_WEIGHTS = False
SAVE_WEIGHTS = True
USE_GAN = True
verbosity = 1

#%%

classes = ["Background", "NCR/NET", "ED", "ET", "WM", "GM", "CSF"]
#classes = ["Background", "NCR/NET", "ED", "ET"]
#classes = ["Non-tumor", "Tumor"]

Nclasses = len(classes)

path = "/nobackup/data/mehfo331/Data/Slices/z/Padded"
image_shape = (256,256,1)

img_path = path + "/t1ce"
mask_path = path  + "/Masks_complete"
#mask_path = path + "/Masks"

img_path_GAN = img_path + "/GAN_Preprocessed/Kept"
mask_path_GAN = mask_path + "/GAN_Preprocessed/Kept"

#%%
    
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

PIXEL_MAX = 11356
#PIXEL_MAX = 65504
IMG_MEAN = np.load(img_path + "/image_mean.npy")
IMG_MEAN_WITH_GAN = np.load(img_path + "/image_mean_with_GANs.npy")

#%%

def data_generator(img_path, mask_path, load_size, batch_size, categorical = True,
                   use_GAN = False, img_path_GAN = None, mask_path_GAN = None, 
                   normalization_constant = PIXEL_MAX, img_mean = IMG_MEAN):
    
    img_dirs = [img_path + "/" + s for s in sorted(os.listdir(img_path))]
    mask_dirs = [mask_path + "/" + s for s in sorted(os.listdir(mask_path))]
    
    if use_GAN:
        img_dirs_gan = [img_path_GAN + "/" + s for s in sorted(os.listdir(img_path_GAN))]
        mask_dirs_gan = [mask_path_GAN + "/" + s for s in sorted(os.listdir(mask_path_GAN))]
        
        img_dirs += img_dirs_gan
        mask_dirs += mask_dirs_gan
    
    Ndata = len(img_dirs)
    
    if Ndata != len(mask_dirs):
        raise Exception('The number of images and image masks do not match.')
        
    if load_size % batch_size != 0:
        raise Exception('batch_size does not divide load_size evenly')
        
    j = 0
    k = 0
    
    shuffle = True
    
    img_load = np.zeros((load_size, *image_shape), dtype = 'float16')
    mask_load = np.zeros((load_size, *image_shape), dtype = 'uint8')

    while True:
        
        if k == 0:
            
            # Randomize dataset after epoch:
            
            if shuffle:
                
                dirs = list(zip(img_dirs, mask_dirs))
                random.shuffle(dirs)
                img_dirs, mask_dirs = zip(*dirs)
                
                shuffle = False
            
            # Load images into memory
    
            for i in range(j, j + load_size):
                
                img = Image.open(img_dirs[i % Ndata])
                img_array = img_to_array(img, dtype = 'float16')
                #img_array = np.asarray(img, dtype = 'float16')
                img_load[i - j] = img_array
                
                mask = Image.open(mask_dirs[i % Ndata])
                mask_array = img_to_array(mask, dtype = 'uint8')
                #mask_array = np.asarray(mask, dtype = 'uint8')
                
                if Nclasses == 2:  
                    mask_array = mask_array.copy()
                    mask_array[mask_array != 0] = 1
                
                mask_load[i - j] = mask_array
            
        # Pick out batch from the data loaded into memory
        
        img_batch = img_load[k:(k + batch_size)]
        mask_batch = mask_load[k:(k + batch_size)]
        
        k += batch_size
        
        if k == load_size:
            k = 0
            j += load_size
            
        if j >= Ndata:
            shuffle = True
            j = 0
        
        # Preprocessing:
            
        # Normalize:            
        img_batch = img_batch/normalization_constant
        
        # Center:       
        img_batch -= img_mean
        
        # Categorize:   
        if categorical:
            mask_batch = to_categorical(mask_batch, Nclasses)
        
        yield(img_batch, mask_batch)

#%%
    
import sys

file_dir = os.path.dirname("/nobackup/data/mehfo331/Code/unet_model.py")
sys.path.append(file_dir)
        
from unet_model import Unet

class_weights = np.load(path + "/Class_weights/class_weights_complete.npy")
#class_weights = np.load(path + "/Class_weights/class_weights_incomplete.npy")
#class_weights = np.load(path + "/Class_weights/class_weights_binary.npy")

class_weights_with_GAN = np.load(path + "/Class_weights/complete_with_GANs.npy")

net = Unet(img_size = image_shape,
             Nclasses = Nclasses,
             class_weights = class_weights,
             class_weights_with_GAN = class_weights_with_GAN,
             use_GAN = USE_GAN,
             depth = 5)

#%%

from keras.callbacks import ModelCheckpoint

weight_path = "/nobackup/data/mehfo331/Code/Unet-weights"

if LOAD_WEIGHTS:
    net.model.load_weights(weight_path + "/Saved/U-net_weights_complete.h5")
    #net.model.load_weights(weight_path + "/Saved/U-net_weights_incomplete.h5")
    #net.model.load_weights(weight_path + "/Saved/U-net_weights_binary.h5")
#%%
    
if SAVE_WEIGHTS:    

    callbacks = [ModelCheckpoint(weight_path + "/U-net_weights_GAN_2.h5",
                                 verbose = 1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 monitor = 'val_weighted_dice_loss')]
else:
    callbacks = None


#%%
        
train_img_path = img_path + "/Training"
val_img_path = img_path + "/Validation"
train_mask_path = mask_path + "/Training"
val_mask_path = mask_path + "/Validation"

Ntraining = len(os.listdir(train_img_path))
Nval = len(os.listdir(val_img_path))

if USE_GAN:
    Ntraining += len(os.listdir(img_path_GAN))

batch_size = 8
epochs = 150

steps_per_epoch = np.ceil(Ntraining/batch_size)
validation_steps = np.ceil(Nval/batch_size)

load_size_train = int(steps_per_epoch * batch_size)
load_size_val = int(validation_steps * batch_size)

if USE_GAN:
    train_generator = data_generator(train_img_path, train_mask_path, load_size_train, batch_size,
                                     img_path_GAN = img_path_GAN, mask_path_GAN = mask_path_GAN, use_GAN = True,
                                     normalization_constant = PIXEL_MAX, img_mean = IMG_MEAN_WITH_GAN)
else:
    train_generator = data_generator(train_img_path, train_mask_path, load_size_train, batch_size)
    
val_generator = data_generator(val_img_path, val_mask_path, load_size_val, batch_size)

#%%

history = net.model.fit_generator(train_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs = epochs,
                                  callbacks = callbacks,
                                  validation_data = val_generator,
                                  validation_steps = validation_steps,
                                  shuffle = False,
                                  verbose = verbosity)

#%%

# val_generator = data_generator(val_img_path, val_mask_path, load_size_val, batch_size)

# _ , dice = net.model.evaluate_generator(val_generator, steps = validation_steps, verbose = 1)

# #%%

# val_steps = int(np.ceil(Nval/batch_size))

# Xval = np.zeros((load_size_val, *image_shape), dtype = 'float32')
# Yval = np.zeros((load_size_val, *image_shape), dtype = 'uint8')

# val_generator = data_generator(val_img_path, val_mask_path, load_size_val, batch_size, categorical = False)

# for i in range(val_steps):
#     Xval[(batch_size * i):(batch_size * (i + 1))], Yval[(batch_size * i):(batch_size * (i + 1))] = next(val_generator)
    
# #%%
# val_generator = data_generator(val_img_path, val_mask_path, load_size_val, batch_size)

# Ypred = net.model.predict_generator(val_generator, steps = validation_steps)
# Ypred = np.argmax(Ypred, axis = -1)
# Ypred = np.expand_dims(Ypred, -1)
# #%%

# from matplotlib import pyplot as plt

# sample = 1

# plt.figure(figsize = (12,12))
# plt.subplot(131)
# plt.imshow(Xval[sample,:,:,0], cmap = "gray")
# plt.title('Image')
# plt.subplot(132)
# plt.imshow(Ypred[sample,:,:,0], cmap = "gray")
# plt.title('Segmentation result')
# plt.subplot(133)
# plt.imshow(Yval[sample,:,:,0], cmap = "gray")
# plt.title('Ground truth segmentation')

#%%
