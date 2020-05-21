import os
import shutil

import random
random.seed(1)

from numpy.random import seed
seed(1)

slice_dim = "z"
format = "t1ce"

data_path = "/nobackup/local/mehfo331/Thesis/Data/Slices/" + slice_dim

#%%

MR_path = data_path + "/" + format
seg_path = data_path + "/Masks"
seg_path_complete = data_path + "/Masks_complete"

# -3 for the number of subdirectories ("Train", "Validation" and "Test")
# Make sure these folders exist before you run the code (too lazy to write an exception for this, deal with it)

Ndata = len(os.listdir(MR_path)) - 3

if Ndata != len(os.listdir(seg_path)) - 3:
    raise Exception('The number of images and image masks do not match.')
    
if Ndata != len(os.listdir(seg_path_complete)) - 3:
    raise Exception('The number of masks and complete masks do not match.')
    
sample_indeces = list(range(0, Ndata))
random.shuffle(sample_indeces)

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

def copy_data(src, dst, a, b):
    j = 0
    for i in range(a,b):
        index = sample_indeces[i] 
        shutil.copyfile(src + str(index)+ ".png", dst + str(j) + ".png")
        j += 1

def delete_data(src):
    for i in range(Ndata):
        os.remove(src + str(i) + ".png")

#%%

# MR images

src = MR_path + "/image"
dst = MR_path + "/Training/train_image"

copy_data(src, dst, 0, train_size)

dst = MR_path + "/Validation/val_image"

copy_data(src, dst, train_size, train_size + validation_size)

dst = MR_path + "/Test/test_image"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%

# Segmentation masks (4 classes)

src = seg_path + "/mask"
dst = seg_path + "/Training/train_mask"

copy_data(src, dst, 0, train_size)

dst = seg_path + "/Validation/val_mask"

copy_data(src, dst, train_size, train_size + validation_size)

dst = seg_path + "/Test/test_mask"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%

# Segmentation masks (7 classes)

src = seg_path_complete + "/mask"
dst = seg_path_complete + "/Training/train_mask"

copy_data(src, dst, 0, train_size)

dst = seg_path_complete + "/Validation/val_mask"

copy_data(src, dst, train_size, train_size + validation_size)

dst = seg_path_complete + "/Test/test_mask"

copy_data(src, dst, train_size + validation_size, Ndata)

#%%

delete_data(src)

#%%