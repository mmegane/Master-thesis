import os
import numpy as np

slice_dim = "z"
format = "t1ce"
USE_GANS = True
PIXEL_MAX = 11356

#path = "/nobackup/local/mehfo331/Thesis/Data/Slices/z"
path = "/nobackup/data/mehfo331_suzuki/Data/Slices/" + slice_dim + "/Padded/" + format

#%%

def read_data(path):
    
    imgs = []
    Ndata = len(os.listdir(path))
    
    print("Reading data...")
    
    for i in range(Ndata):
        
        img = Image.open(path + "/" + str(i).zfill(5) + ".png")
        img_array = np.asarray(img, dtype = 'float16')
        imgs.append(img_array)
        
    imgs = np.asarray(imgs, dtype = 'float16')
        
    print("Finished reading data.")
    
    return(imgs)
    

#%%

from PIL import Image

imgs = read_data(path + "/Training")
imgs = imgs/PIXEL_MAX

imgs_mean = np.mean(imgs, axis = (0,1,2))

#%%

if USE_GANS:
    
    imgs_GAN = read_data(path + "/GAN_Preprocessed/Kept")
    imgs_GAN = imgs_GAN/PIXEL_MAX
    
    imgs_total = np.concatenate((imgs, imgs_GAN[0:4955]), axis = 0)
    image_mean_total = np.mean(imgs_total, axis = (0,1,2))

#%%

#np.save(path + "/imgs_mean.npy", imgs_mean)

if USE_GANS:
    np.save(path + "/image_mean_with_GANs_(5k).npy", image_mean_total)
    