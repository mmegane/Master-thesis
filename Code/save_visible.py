import numpy as np
from PIL import Image
import os
import matplotlib.colors
import matplotlib.pyplot as plt

in_dir = "/nobackup/data/mehfo331/Data/Slices/z/Masks_complete/Test"
out_dir = "/nobackup/data/mehfo331"

N = 3

dtype = 'uint8'

#classes = ["Background", "NCR/NET", "ED", "ET", "WM", "GM", "CSF"]
# BG = black, CSF = blue, GM = grey, WM = white, ED = yellow, NCR = red, ET = orange

colors = ['black', 'white']
#colors = ['black', '#ff0000', '#ffe517', '#ff8b17']
#colors = ['black', '#2389da','#aeaeae', 'white', '#ff0000', '#ffe517', '#ff8b17']

#%%

def load_array(dir, N, dtype):
    dir_list = sorted(os.listdir(dir))
    dir_list = [s for s in dir_list if '.png' in s]

    img_tensor = np.zeros((N, 256,256), dtype = dtype)

    for i in range(N):
        
        image = Image.open(dir + "/" + dir_list[i])
        image = np.asarray(image, dtype = dtype)
        
        image = image.copy()
        
        # image[(image >= 1) & (image <= 3)] = 0
        # image[image == 4] = 1
        # image[image == 5] = 2
        # image[image == 6] = 3
        
        image[(image >= 1) & (image <= 3)] = 0
        image[image == 4] = 1
        image[image == 5] = 1
        image[image == 6] = 1
    
        img_tensor[i] = image
        
    return(img_tensor)

def save_array(array, N, dir, colors = None):
    
    for i in range(N):
        
        slice = array[i]
        
        Nclasses = len(np.unique(slice))
        cmap = matplotlib.colors.ListedColormap(colors, N = Nclasses)
        
        plt.imsave(dir + "/" + str(i).zfill(5) + ".png", slice, cmap = cmap)
        
#%%

array = load_array(in_dir, N, dtype)
save_array(array, N, out_dir, colors)
