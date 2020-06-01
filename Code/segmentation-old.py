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

classes = ["Background", "NCR/NET", "ED", "ET", "WM", "GM", "CSF"]
#classes = ["Background", "NCR/NET", "ED", "ET"]
#classes = ["Non-tumor", "Tumor"]

Nclasses = len(classes)

path = "/nobackup/data/mehfo331_suzuki"
data_path = path + "/Data/Slices"
code_path = path + "/Code"

# img_path = data_path + "/z/Original/t1ce"
# mask_path = data_path  + "/z/Original/Masks_complete"
# #mask_path = data_path + "/z/Original/Masks"

# image_shape = (240,240,1)

img_path = data_path + "/z/Padded/t1ce"
mask_path = data_path  + "/z/Padded/Masks_complete"
#mask_path = data_path + "/z/Padded/Masks"

image_shape = (256,256,1)
        
#%%
    
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

PIXEL_MAX = 11356
IMG_MEAN = np.load(img_path + "/image_mean.npy")

INCOMPLETE = True

def data_generator(img_path, mask_path, load_size, batch_size, categorical = True):
    img_dir = sorted(os.listdir(img_path))
    mask_dir = sorted(os.listdir(mask_path))
    
    Ndata = len(img_dir)
    
    if Ndata != len(mask_dir):
        raise Exception('The number of images and image masks do not match.')
        
    if load_size % batch_size != 0:
        raise Exception('batch_size does not divide load_size evenly')
        
    j = 0
    k = 0
    
    shuffle = False
    
    img_load = np.zeros((load_size, *image_shape), dtype = 'float16')
    mask_load = np.zeros((load_size, *image_shape), dtype = 'uint8')

    while True:
        
        if k == 0:
            
            # Randomize dataset after epoch:
            
            if shuffle:
                
                dirs = list(zip(img_dir, mask_dir))
                random.shuffle(dirs)
                img_dir, mask_dir = zip(*dirs)
                
                shuffle = False
            
            # Load images into memory
    
            for i in range(j, j + load_size):
                
                img = Image.open(img_path + "/" + img_dir[i % Ndata])
                img_array = img_to_array(img, dtype = 'float16')
                #img_array = np.asarray(img, dtype = 'float16')
                img_load[i - j] = img_array
                
                mask = Image.open(mask_path + "/" + mask_dir[i % Ndata])
                mask_array = img_to_array(mask, dtype = 'uint8')
                #mask_array = np.asarray(mask, dtype = 'uint8')
                
                if Nclasses == 2:  
                    mask_array = mask_array.copy()
                    mask_array[mask_array != 0] = 1
                
                elif Nclasses == 4:
                    mask_array = mask_array.copy()
                    mask_array[mask_array == 4] = 3                              
                
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
        img_batch = img_batch/PIXEL_MAX
        
        # Center:
        img_batch -= IMG_MEAN
        
        # Categorize:
        if categorical:
            mask_batch = to_categorical(mask_batch, Nclasses)
        
        yield(img_batch, mask_batch)

#%%
    
import sys

file_dir = os.path.dirname(code_path + "/unet_model_old.py")
sys.path.append(file_dir)
        
from unet_model_old import Unet

#class_weight_path = data_path + "/z/Original/Class_weights"
class_weight_path = data_path + "/z/Padded/Class_weights"

class_weights = np.load(class_weight_path + "/class_weights_complete.npy")
#class_weights = np.load(class_weight_path + "/class_weights_incomplete.npy")
#class_weights = np.load(class_weight_path + "/class_weights_binary.npy")

net = Unet(img_size = image_shape,
             Nclasses = Nclasses,
             class_weights = class_weights,
             depth = 5)

#%%

from keras.callbacks import ModelCheckpoint

model_weight_path = code_path + "/Unet-weights"

#net.model.load_weights(model_weight_path + "/Saved/U-net_weights_complete.h5")
#net.model.load_weights(model_weight_path + "/Saved/U-net_weights_incomplete.h5")
#net.model.load_weights(model_weight_path + "/Saved/U-net_weights_binary.h5")

#%%

callbacks = [ModelCheckpoint(model_weight_path + "/test.h5",
                             verbose = 1,
                             save_best_only = True,
                             save_weights_only = True)]

                             #EarlyStopping(patience = 15)]

#callbacks = None
#%%
        
train_img_path = img_path + "/Training"
val_img_path = img_path + "/Validation"
train_mask_path = mask_path + "/Training"
val_mask_path = mask_path + "/Validation"

Ntraining = len(os.listdir(train_img_path))
Nval = len(os.listdir(val_img_path))

batch_size = 8
epochs = 150

steps_per_epoch = np.ceil(Ntraining/batch_size)
validation_steps = np.ceil(Nval/batch_size)

load_size_train = int(steps_per_epoch * batch_size)
load_size_val = int(validation_steps * batch_size)

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
                                  verbose = 2)

# #%%

# val_generator = data_generator(val_img_path, val_mask_path, load_size_val, batch_size)

# dice, acc = net.model.evaluate_generator(val_generator, steps = validation_steps, verbose = 1)

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

# sample = 1000

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
