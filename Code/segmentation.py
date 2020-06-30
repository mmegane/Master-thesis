seed = 0

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

#%%

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

image_shape = (256,256,1)

path = "/nobackup/data/mehfo331/Data/Slices/z/New"
#path = "../Data/Slices/z/New"

img_path = path + "/t1ce"
mask_path = path  + "/Masks_complete"
#mask_path = path + "/Masks"

#img_path_train = img_path + "/Training"
img_path_train = img_path + "/Training/Fifth"
img_path_val = img_path + "/Validation"

#img_path_val = "/nobackup/data/mehfo331/Data/Slices/z/Old/t1ce/Validation"

#mask_path_train = mask_path + "/Training"
mask_path_train = mask_path + "/Training/Fifth"
mask_path_val = mask_path + "/Validation"

#mask_path_val = "/nobackup/data/mehfo331/Data/Slices/z/Old/Masks_complete/Validation"

img_path_GAN = img_path + "/GAN_Preprocessed/Kept"
mask_path_GAN = mask_path + "/GAN_Preprocessed/Kept"

#%%

BATCH_SIZE = 8
EPOCHS = 150

VERBOSITY = 1

LOAD_WEIGHTS = False
LOAD_NAME = None

SAVE_WEIGHTS = False

TRAIN_RATIO = 1
GAN_RATIO = 0

TEST = False

#PIXEL_MAX = 11360
#PIXEL_MAX = 11356
#PIXEL_MAX = 65535

#%%

from PIL import Image

def return_img_paths(path, ratio = 1):
    
    paths = sorted(os.listdir(path))
    Ndata = np.rint(ratio * len(paths)).astype(np.int32)
    paths_complete = [path + "/" + paths[i] for i in range(0, Ndata)]
     
    return(paths_complete)

def return_img_tensor(path_1, path_2 = None,
                      img_shape = (256,256),
                      ratio_1 = 1, ratio_2 = 0,
                      dtype = 'float16'):
    
    dirs_1 = return_img_paths(path_1, ratio_1)
    
    if ratio_2 != 0:
        dirs_2 = return_img_paths(path_2, ratio_2)
    else:
        dirs_2 = []
        
    dirs_final = dirs_1 + dirs_2
    
    Ndata = len(dirs_final)
    img_tensor = np.zeros((Ndata, *img_shape), dtype = dtype)
    
    print("\nReading data...")
    
    for i in range(Ndata):
        img = Image.open(dirs_final[i])
        img_array = np.asarray(img, dtype = dtype)
        img_tensor[i] = img_array
        
    if dtype == 'uint8':
        if Nclasses == 2:  
            img_tensor = img_tensor.copy()
            img_tensor[img_tensor != 0] = 1
                    
        elif Nclasses == 4:
            img_tensor = img_tensor.copy()
            img_tensor[img_tensor == 4] = 3 
        
    print("Finished reading data.")
        
    return(img_tensor)

def return_class_weights(Y):
    
    eps = 1e-5
    num = Y.shape[0]*Y.shape[1]*Y.shape[2]
    den = Nclasses * np.unique(Y, return_counts = True)[1] + eps
    
    weights = num/den
    
    return(weights)

#%%
    
# Calculate means and class weights
    
img_tensor_train = return_img_tensor(img_path_train, img_path_GAN,
                                     ratio_1 = TRAIN_RATIO, ratio_2 = GAN_RATIO, dtype = 'float16')

mask_tensor_train = return_img_tensor(mask_path_train, mask_path_GAN,
                                      ratio_1 = TRAIN_RATIO, ratio_2 = GAN_RATIO, dtype ='uint8')

pixel_max_train = int(np.max(img_tensor_train))
img_mean_train = np.mean(img_tensor_train/pixel_max_train, axis = (0,1,2))
class_weights_train = return_class_weights(mask_tensor_train)

#del(img_tensor_train)
#del(mask_tensor_train)

img_tensor_val = return_img_tensor(img_path_val, ratio_1 = TRAIN_RATIO, dtype = 'float16')
mask_tensor_val = return_img_tensor(mask_path_val, ratio_1 = TRAIN_RATIO, dtype = 'uint8')

pixel_max_val = int(np.max(img_tensor_val))
img_mean_val = np.mean(img_tensor_val/pixel_max_train, axis = (0,1,2))
class_weights_val = return_class_weights(mask_tensor_val)

#%%
    
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

def data_generator(img_path, mask_path, load_size, batch_size, categorical = True,
                    img_path_GAN = None, mask_path_GAN = None, 
                    normalization_constant = pixel_max_train, img_mean = img_mean_train,
                    real_ratio = 1.0, GAN_ratio = 0):
    
    img_dirs = return_img_paths(img_path, ratio = real_ratio)
    mask_dirs = return_img_paths(mask_path, ratio = real_ratio)
    
    if GAN_ratio > 0:   
        img_dirs += return_img_paths(img_path_GAN, ratio = GAN_ratio)
        mask_dirs += return_img_paths(mask_path_GAN, ratio = GAN_ratio)
    
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
                    
                elif Nclasses == 4:
                    mask_array = mask_array.copy()
                    mask_array[mask_array == 4] = 3  
                
                mask_load[i - j] = mask_array
        
            
        # Pick out batch from the data loaded into memory*
        
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

file_dir = os.path.dirname("unet_model_old.py")
sys.path.append(file_dir)
        
from unet_model_old import Unet

net = Unet(img_size = image_shape,
              Nclasses = Nclasses,
              class_weights = class_weights_train,
              depth = 5)
        
#%%
    
# import sys

# file_dir = os.path.dirname("unet_model.py")
# sys.path.append(file_dir)
        
# from unet_model import Unet

# net = Unet(img_size = image_shape,
#               Nclasses = Nclasses,
#               class_weights_train = class_weights_train,
#               class_weights_val = class_weights_train,
#               depth = 5)
#%%

Nreal = np.rint(TRAIN_RATIO * len(os.listdir(img_path_train))).astype(np.int32)
#NGan = np.rint(GAN_RATIO * len(os.listdir(img_path_GAN))).astype(np.int32)
NGan = 0
Nval = len(os.listdir(img_path_val))

Ntraining = Nreal + NGan

batch_size = BATCH_SIZE
epochs = EPOCHS

steps_per_epoch = np.ceil(Ntraining/batch_size)
validation_steps = np.ceil(Nval/batch_size)

load_size_train = int(steps_per_epoch * batch_size)
load_size_val = int(validation_steps * batch_size)

train_generator = data_generator(img_path_train, mask_path_train, load_size_train, batch_size,
                                  img_path_GAN = img_path_GAN, mask_path_GAN = mask_path_GAN,
                                  normalization_constant = pixel_max_train, img_mean = img_mean_train,
                                  real_ratio = TRAIN_RATIO, GAN_ratio = GAN_RATIO)

val_generator = data_generator(img_path_val, mask_path_val, load_size_val, batch_size,
                                normalization_constant = pixel_max_train, img_mean = img_mean_train)

#%%

from keras.callbacks import ModelCheckpoint

weight_path = "./Unet-weights/New"

if LOAD_WEIGHTS:
    net.model.load_weights(weight_path + "/Saved/" + LOAD_NAME)
    
if SAVE_WEIGHTS:

    save_name = str(Nclasses) + "_classes_" + str(Nreal) + "_reals_" + str(NGan) + "_GANs.h5"

    callbacks = [ModelCheckpoint(weight_path + "/" + save_name,
                                 verbose = 1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 monitor = 'val_weighted_dice_loss')]
else:
    callbacks = None

#%%

print("\nTraining " + str(Nclasses) + " classes with " + str(Ntraining) + " images (" + str(Nreal) + " real and " + str(NGan) + " GAN images):\n")

history = net.model.fit_generator(train_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs = epochs,
                                  callbacks = callbacks,
                                  validation_data = val_generator,
                                  validation_steps = validation_steps,
                                  shuffle = False,
                                  verbose = VERBOSITY)

print("Finished training.")

#%%

if TEST:

    val_generator = data_generator(img_path_val, mask_path_val, load_size_val, batch_size)
    
    _ , dice = net.model.evaluate_generator(val_generator, steps = validation_steps, verbose = 1)
    
    #%%
    
    val_steps = int(np.ceil(Nval/batch_size))
    
    Xval = np.zeros((load_size_val, *image_shape), dtype = 'float32')
    Yval = np.zeros((load_size_val, *image_shape), dtype = 'uint8')
    
    val_generator = data_generator(img_path_val, mask_path_val, load_size_val, batch_size, categorical = False)
    
    for i in range(val_steps):
        Xval[(batch_size * i):(batch_size * (i + 1))], Yval[(batch_size * i):(batch_size * (i + 1))] = next(val_generator)
        
    #%%
    val_generator = data_generator(img_path_val, mask_path_val, load_size_val, batch_size)
    
    Ypred = net.model.predict_generator(val_generator, steps = validation_steps)
    Ypred = np.argmax(Ypred, axis = -1)
    Ypred = np.expand_dims(Ypred, -1)
    #%%
    
    from matplotlib import pyplot as plt
    
    sample = 1
    
    plt.figure(figsize = (12,12))
    plt.subplot(131)
    plt.imshow(Xval[sample,:,:,0], cmap = "gray")
    plt.title('Image')
    plt.subplot(132)
    plt.imshow(Ypred[sample,:,:,0], cmap = "gray")
    plt.title('Segmentation result')
    plt.subplot(133)
    plt.imshow(Yval[sample,:,:,0], cmap = "gray")
    plt.title('Ground truth segmentation')

#%%
