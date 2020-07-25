import os
import shutil

import random
random.seed(1)

from numpy.random import seed
seed(1)

slice_dim = "z"
format = "t1ce"

#data_path = "/nobackup/local/mehfo331/Data/Slices/" + slice_dim
data_path = "/nobackup/data/mehfo331/Data/Slices/" + slice_dim + "/Padded"

#%%

MR_path = data_path + "/" + format
seg_path = data_path + "/Masks"
seg_path_complete = data_path + "/Masks_complete"

# This script is meant to be run directly after 'slicer.py'
# -3 for the number of subdirectories ("Train", "Validation" and "Test")
# Make sure these folders exist before you run the code (too lazy to write an exception for this, deal with it)

Ndata = len(os.listdir(MR_path)) - 3

if Ndata != len(os.listdir(seg_path)) - 3:
    raise Exception('The number of images and image masks do not match.')
    
if Ndata != len(os.listdir(seg_path_complete)) - 3:
    raise Exception('The number of masks and complete masks do not match.')
    
#%%

# These values are hard coded and calculated ahead of time. Why? Too many edge cases.
# For reference:
# N = 210*155 = 32550
# train_ratio = 0.8
# validation_ratio = 0.1
# test_ratio = 0.1

train_size = 26040
validation_size = 3255
test_size = 3255

#%%

sample_indeces = list(range(0, Ndata))
random.shuffle(sample_indeces)

#%%

def copy_data(src, dst, a = 0, b = None, shuffle = True):
    j = 0
    
    dir_list = sorted(os.listdir(src))
    dir_list = [s for s in dir_list if '.png' in s]
    
    if b == None:
        b = len(dir_list)
    
    for i in range(a,b):
        
        if shuffle:
            index = sample_indeces[i]
        else:
            index = j
        
        #shutil.copyfile(src + "/" + str(index).zfill(5) + ".png", dst + "/" + str(j).zfill(5) + ".png")
        shutil.copyfile(src + "/" + dir_list[index], dst + "/" + str(j).zfill(5) + ".png")
        j += 1

def delete_data(src):
    Ndata = len(os.listdir(src))
    for i in range(Ndata):
        os.remove(src + "/" + str(i).zfill(5) + ".png")
    
def rename_files(src):
    files = sorted(os.listdir(src))
    for i in range(len(files)):
        os.rename(src + "/" + files[i], src + "/" + str(i).zfill(5) + ".png")
        

#%%

# MR images

src = MR_path

dst = MR_path + "/Training"

copy_data(src, dst, 0, train_size, shuffle = True)

dst = MR_path + "/Validation"

copy_data(src, dst, train_size, train_size + validation_size)

dst = MR_path + "/Test"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%

# Segmentation masks (4 classes)

src = seg_path
dst = seg_path + "/Training"

copy_data(src, dst, 0, train_size)

dst = seg_path + "/Validation"

copy_data(src, dst, train_size, train_size + validation_size)

dst = seg_path + "/Test"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%

# Segmentation masks (7 classes)

src = seg_path_complete + "/"
dst = seg_path_complete + "/Training/"

copy_data(src, dst, 0, train_size)

dst = seg_path_complete + "/Validation/"

copy_data(src, dst, train_size, train_size + validation_size)

dst = seg_path_complete + "/Test/"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%

# MR_images (GAN):

src = "/nobackup/data/mehfo331/Code/SPADE/results/PGAN-brains_full_useable/PGAN-brains_full/test_latest/images/synthesized_image"
dst = "/nobackup/data/mehfo331/Data/Slices/z/t1ce/GAN/Full/Preprocessed/Kept"

Ndata = len(os.listdir(src))

copy_data(src, dst, shuffle = False)

#%%

# Segmentation masks (7 classes, GAN)

src = "/nobackup/data/mehfo331/Code/progressive_growing_of_gans/results/006-fake-images-2"
dst = "/nobackup/data/mehfo331/Data/Slices/z/Masks_complete/GAN/Full/Raw"

copy_data(src, dst, shuffle = False)

#%%

# Rename files

src = seg_path + "/Training"

rename_files(src)