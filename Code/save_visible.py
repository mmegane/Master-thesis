import numpy as np
from PIL import Image
import os
import matplotlib.colors
import matplotlib.pyplot as plt

import png

# in_dir = "/nobackup/data/mehfo331/Data/Slices/z/Masks/Test"
# out_dir = "/nobackup/data/mehfo331/temp/2/from_4"

# N = 200

# dtype = 'uint8'

#classes = ["Background", "NCR/NET", "ED", "ET", "WM", "GM", "CSF"]
# BG = black, CSF = blue, GM = grey, WM = white, NCR = red, ED = yellow, ET = orange

#colors = ['black', 'white']
#olors = ['black', '#ff0000', '#ffe517', '#ff8b17']
#colors = ['black', '#2389da','#aeaeae', 'white', '#ff0000', '#ffe517', '#ff8b17']

#Nclasses = 2

color_table_2 = {0: [0,0,0],
                 1: [255, 255, 255]}

color_table_4 = {0: [0,0,0],
                 1: [255, 0, 0],
                 2: [255, 229, 23],
                 3: [255, 139, 23]}

color_table_7 = {0: [0,0,0],
                 1: [35, 137, 228],
                 2: [174, 174, 174],
                 3: [255, 255, 255],
                 4: [255, 0, 0],
                 5: [255, 229, 23],
                 6: [255, 139, 23]}

classes_to_table = {2: color_table_2, 4: color_table_4, 7: color_table_7}
#%%

def load_array(dir, N, dtype):
    dir_list = sorted(os.listdir(dir))
    dir_list = [s for s in dir_list if '.png' in s]

    img_tensor = np.zeros((N, 256,256), dtype = dtype)

    for i in range(N):
        
        image = Image.open(dir + "/" + dir_list[i])
        image = np.asarray(image, dtype = dtype)
    
        img_tensor[i] = image
        
    return(img_tensor)

def save_array(array, N, dir, colors = None):
    
    for i in range(N):
        
        slice = array[i]
        
        Nclasses = len(np.unique(slice))
        cmap = matplotlib.colors.ListedColormap(colors, N = Nclasses)
        
        plt.imsave(dir + "/" + str(i).zfill(5) + ".png", slice, cmap = cmap)

#%%

# def convert_to_color(array, color_table):
#     N = array.shape[0]
    
#     color_array = np.zeros((N, 256, 256*3), dtype = 'uint8')
    
#     for i in range(N):
        
#         array_i = array[i]
        
#         for j in range(256):
#             for k in range(256):
                
#                 label = array_i[j,k]
#                 color = color_table[label]
                
#                 color_array[i, j, k] = color[0]
#                 color_array[i, j, k + 1] = color[1]
#                 color_array[i, j, k + 2] = color[2]
        
        
#     return(color_array)
        
def convert_to_color(array, N, Nclasses):
    
    array = array[0:N]
    array = np.reshape(array, (N, 256*256))
    color_array = np.zeros((N, 256 * 256 * 3), dtype = 'uint8')
    
    color_table = classes_to_table[Nclasses]
    
    for i in range(N):
        vector_i = array[i]
        
        color_index = 0
        for j in range(256*256):
            
            label = vector_i[j]
            color = color_table[label]
            
            color_array[i, color_index] = color[0]
            color_array[i, color_index + 1] = color[1]
            color_array[i, color_index + 2] = color[2]
            
            color_index += 3
            
    color_array = np.reshape(color_array, (N, 256, 256*3))
    return(color_array)

def save_color_arrays(array, N, dir):
    
    for i in range(N):
        
        slice = array[i]
        
        f = open(dir + "/" + str(i).zfill(5) + ".png", 'wb')
        w = png.Writer(256, 256, greyscale = False)
        w.write(f, slice)
        f.close()
#%%

# array = load_array(in_dir, N, dtype)
# color_array = convert_to_color(array, Nclasses)
# #save_array(array, N, out_dir, colors)
# save_color_arrays(color_array, N, out_dir)
