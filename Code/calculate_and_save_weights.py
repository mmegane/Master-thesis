import os
import numpy as np

classes = ["Background", "NCR/NET", "ED", "ET", "WM", "GM", "CSF"]
#classes = ["Background", "NCR/NET", "ED", "ET"]
#classes = ["Non-tumor", "Tumor"]

class_type = "complete"
class_dir = {"binary": "Masks", "incomplete": "Masks", "complete": "Masks_complete"}


Nclasses = len(classes)

slice_dim = "z"
USE_GANS = False

#path = "/nobackup/data/mehfo331/Data/Slices/" + slice_dim
path = "/nobackup/data/mehfo331/Data/Slices/" + slice_dim + "/Padded"

# if Ndata != len(os.listdir(path + "/Masks/Training")):
#     raise Exception('The number of images and image masks do not match.')


#%%

from PIL import Image

def read_data(path):

    print("Reading data...")
    Ndata = len(os.listdir(path))

    masks = np.zeros((Ndata, 256, 256), dtype = 'uint8')
    
    for i in range(Ndata):
        
        mask = Image.open(path + "/" + str(i).zfill(5) + ".png")
        mask_array = np.asarray(mask, dtype = 'uint8')
        
        if Nclasses == 2:
            mask_array = mask_array.copy()
            mask_array[mask_array != 0] = 1
            
        #masks.append(mask_array)
        #masks = np.asarray(masks, dtype = 'uint8')
            
        masks[i] = mask_array
        
    print("Finished reading data.")
        
    return(masks)
        

#%%

masks = read_data(path + "/" + class_dir[class_type] + "/Training")

if USE_GANS:
    masks_GAN = read_data(path + "/" + class_dir[class_type] + "/GAN_Preprocessed/Kept")

#%%

def calculate_class_weights(Y):
    eps = 1e-5
    
    num = Y.shape[0]*Y.shape[1]*Y.shape[2]
    den = Nclasses * np.unique(Y, return_counts = True)[1] + eps
    
    return(num/den)

class_weights = calculate_class_weights(masks)

if USE_GANS:
    masks_total = np.concatenate((masks, masks_GAN[0:4955]), axis = 0)
    class_weights_total = calculate_class_weights(masks_total)

#%%

#np.save(path + "/Class_weights/" + class_type + ".npy", class_weights)

if USE_GANS:
    np.save(path + "/Class_weights/" + class_type + "_with_GANs_(5k)" + ".npy", class_weights_total)

#np.save(path + "Class_weights/class_weights_complete.npy", class_weights)
#np.save(path + "Class_weights/class_weights_incomplete.npy", class_weights)
#np.save(path + "Class_weights/class_weights_binary.npy", class_weights)
    

#%%