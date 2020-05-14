import os
import numpy as np

#classes = ["Background", "NCR/NET", "ED", "ET", "WM", "GM", "CSF"]
#classes = ["Background", "NCR/NET", "ED", "ET"]
classes = ["Non-tumor", "Tumor"]

Nclasses = len(classes)

#path = "/nobackup/data/mehfo331/Thesis/Data/Slices/z"
path = "/nobackup/data/mehfo331/Thesis/Data/Slices/z/Padded/"

Ndata = len(os.listdir(path + "/Masks/Training"))

if Ndata != len(os.listdir(path + "/Masks/Training")):
    raise Exception('The number of images and image masks do not match.')


#%%

from PIL import Image
from keras.preprocessing.image import img_to_array

masks = []

print("---------------")
print("Reading data...")
print("---------------")

for i in range(Ndata):
    
    #mask = Image.open(path + "/Masks_complete/Training/train_mask" + str(i).zfill(5) + ".png")
    mask = Image.open(path + "/Masks/Training/train_mask" + str(i).zfill(5) + ".png")
    
    mask_array = img_to_array(mask, dtype = 'uint8')
    
    if Nclasses == 2:
        mask_array = mask_array.copy()
        mask_array[mask_array != 0] = 1
    
    masks.append(mask_array)
    
print("Finished reading data.")
print("---------------")

masks = np.asarray(masks, dtype = 'uint8')

#%%

def calculate_class_weights(Y):
    eps = 1e-5
    
    num = Y.shape[0]*Y.shape[1]*Y.shape[2]
    den = Nclasses * np.unique(Y, return_counts = True)[1] + eps
    
    return(num/den)

class_weights = calculate_class_weights(masks)

#%%

#np.save(path + "Class_weights/class_weights_complete.npy", class_weights)
#np.save(path + "Class_weights/class_weights_incomplete.npy", class_weights)
np.save(path + "Class_weights/class_weights_binary.npy", class_weights)

#%%