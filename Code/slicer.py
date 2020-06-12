import os
import numpy as np
import nibabel as nib
import png
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

slice_dim = "z"
dim_dict = {"x": 0, "y": 1, "z": 2}

format = "t1ce"

PADDING = True
PAD_WIDTH = 8

# data_path = "/nobackup/local/mehfo331/Data/Brats"
# output_path = "/nobackup/local/mehfo331/Data/Slices/"

# complete_mask_path = "/nobackup/local/mehfo331/Thesis/Data/CompleteTargetsOrient_RAI/HGG"

data_path = "/nobackup/data/mehfo331/Data/Brats"
output_path = "/nobackup/data/mehfo331/Data/Slices/"

complete_mask_path = "/nobackup/data/mehfo331/Data/CompleteTargetsOrient_RAI/HGG"

#%%

# MR images

file_list = []
for (root, dirs, files) in os.walk(data_path):

    match = [s for s in files if format + ".nii.gz" in s]
    if (len(match) > 0):
        file_list.append(root + "/" + match[0])

file_list = sorted(file_list)

print("Slicing MR images...")

j = 0
for dir in file_list:

    img = nib.load(dir)
    img_data = img.get_fdata()
    
    axis = dim_dict[slice_dim]
    
    slice_count = img_data.shape[axis]
    
    for i in range(slice_count):
        slice = np.take(img_data, i, axis)
        slice = slice.astype('uint16')
        
        if PADDING:
            slice = np.pad(slice, pad_width = PAD_WIDTH, constant_values = 0)           
        
        path = output_path + "/" + slice_dim + "/New/" + format + "/"+ str(j).zfill(5) + ".png"
        
        png.from_array(slice, mode = 'L' + ';16').save(path)
        
        j += 1
        
print("Finished slicing MR images.")
print("\n")

#%%
      
# Segmentation masks (4 classes)       
        
mask_list = []
for (root, dirs, files) in os.walk(data_path):

    match = [s for s in files if "seg.nii.gz" in s]
    if (len(match) > 0):
        mask_list.append(root + "/" + match[0])
        
mask_list = sorted(mask_list)

print("Slicing image masks...")

j = 0
for dir in mask_list:

    img = nib.load(dir)
    img_data = img.get_fdata()
    
    axis = dim_dict[slice_dim]
    
    slice_count = img_data.shape[axis]
    
    for i in range(slice_count):
        slice = np.take(img_data, i, axis)
        slice = slice.astype('uint8')
        
        if PADDING:
            slice = np.pad(slice, pad_width = PAD_WIDTH, constant_values = 0)    
            
        path = output_path + "/" + slice_dim + "/New/Masks/"+ str(j).zfill(5) + ".png" 
        
        png.from_array(slice.copy(), mode = 'L' + ';8').save(path)
        
        j += 1
        
print("Finished slicing image masks.")
print("\n")
    
#%%

# Segmentation masks (7 classes)

mask_list = []
for (root, dirs, files) in os.walk(complete_mask_path):

    mask_list = [root + "/" + s for s in files]

mask_list = sorted(mask_list)

print("Slicing (complete) image masks...")
print("\n")

j = 0
for dir in mask_list:

    img = nib.load(dir)
    img_data = img.get_fdata()
    
    axis = dim_dict[slice_dim]
    
    slice_count = img_data.shape[axis]
    
    for i in range(slice_count):
        slice = np.take(img_data, i, axis)
        slice = slice.astype('uint8')
        
        if PADDING:
            slice = np.pad(slice, pad_width = PAD_WIDTH, constant_values = 0)     
        
        path = output_path + "/" + slice_dim + "/New/Masks_complete/" + str(j).zfill(5) + ".png" 
        
        png.from_array(slice.copy(), mode = 'L' + ';8').save(path)
        
        j += 1
        
print("Finished slicing (complete) image masks.")
print("\n")