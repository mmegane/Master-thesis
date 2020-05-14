import os
import numpy as np

slice_dim = "z"
format = "t1ce"

#path = "/nobackup/local/mehfo331/Thesis/Data/Slices/z"
path = "/nobackup/data/mehfo331/Thesis/Data/Slices/" + slice_dim + "/Padded/"

Ndata = len(os.listdir(path + "/t1ce/Training"))

#%%

from PIL import Image
from keras.preprocessing.image import img_to_array

imgs = []

print("---------------")
print("Reading data...")
print("---------------")

for i in range(Ndata):
    
    img = Image.open(path + + format + "/Training/train_image" + str(i).zfill(5) + ".png")
    img_array = img_to_array(img, dtype = 'float16')
    imgs.append(img_array)
    
print("Finished reading data.")
print("---------------")

#%%

PIXEL_MAX = 11356

imgs = np.asarray(imgs, dtype = 'float16')
imgs = imgs/PIXEL_MAX

#%%
image_mean = np.mean(imgs, axis = (0,1,2))
#%%

np.save(path + format + "/image_mean.npy", image_mean)