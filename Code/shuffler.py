import os
import shutil

import random
#random.seed(1)

slice_dim = "z"
format = "t1ce"

data_path = "/nobackup/data/mehfo331/Data/Slices/" + slice_dim + "/New"

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
train_size_new = 5208
validation_size = 3255
test_size = 3255

#%%

def copy_data(src, dst, a = 0, b = None, shuffle = True, seed = 0):
    
    dir_list = sorted(os.listdir(src))
    dir_list = [s for s in dir_list if '.png' in s]
    
    if b == None:
        b = len(dir_list)
        
    sample_indeces = list(range(a, b))
    
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(sample_indeces)
    
    Ndata = len(sample_indeces)
    
    print(sample_indeces)
    
    for i in range(Ndata):
        
        index = sample_indeces[i] 
        shutil.copyfile(src + "/" + dir_list[index], dst + "/" + str(i).zfill(5) + ".png")

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
dst = MR_path + "/Training/Full"
dst_new = MR_path + "/Training/Fifth"

copy_data(src, dst, 0, train_size)
copy_data(src, dst_new, 0, train_size_new)

dst = MR_path + "/Validation"

copy_data(src, dst, train_size, train_size + validation_size)

dst = MR_path + "/Test"

copy_data(src, dst, train_size + test_size, Ndata)

#%%

delete_data(src)

#%%

# Segmentation masks (4 classes)

src = seg_path
dst = seg_path + "/Training/Full"
dst_new = seg_path + "/Training/Fifth"

copy_data(src, dst, 0, train_size)
copy_data(src, dst_new, 0, train_size_new)

dst = seg_path + "/Validation"

copy_data(src, dst, train_size, train_size + validation_size)

dst = seg_path + "/Test"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%

# Segmentation masks (7 classes)

src = seg_path_complete + "/"
dst = seg_path_complete + "/Training/Full"
dst_new = seg_path_complete + "/Training/Fifth"

copy_data(src, dst, 0, train_size)
copy_data(src, dst_new, 0, train_size_new)

dst = seg_path_complete + "/Validation/"

copy_data(src, dst, train_size, train_size + validation_size)

dst = seg_path_complete + "/Test/"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%

# MR_images (GAN):

src = "/nobackup/data/mehfo331/Code/SPADE/results/PGAN-brains2/test_latest/images/synthesized_image"
dst = "/nobackup/data/mehfo331/Data/Slices/z/Padded/t1ce/GAN_Preprocessed/Kept"

Ndata = len(os.listdir(src))

copy_data(src, dst, shuffle = False)

#%%

# Segmentation masks (7 classes, GAN)

src = "/nobackup/data/mehfo331/Code/progressive_growing_of_gans/results/051-fake-images-47"
dst = "/nobackup/data/mehfo331/Data/Slices/z/Padded/Masks_complete/GAN"

copy_data(src, dst, shuffle = False)

#%%

# Rename files

src = seg_path + "/Training"

rename_files(src)