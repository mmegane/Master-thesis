import numpy as np
from PIL import Image
import png
import os

in_dir = "/nobackup/data/mehfo331/Data/Slices/z/Masks_complete/Training/Fifth"
#out_dir = "/nobackup/data/mehfo331/Data/Slices/temp/test"

# Full
#PIXEL_MAX = 11360
# Fifth
#PIXEL_MAX = 11136

#PIXEL_MAX = 6

#%%

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def load_array(dir):
    dir_list = sorted(os.listdir(dir))
    dir_list = [s for s in dir_list if '.png' in s]
    
    N = len(dir_list)

    img_tensor = np.zeros((N, 256,256), dtype = 'uint8')

    for i in range(N):
        
        image = Image.open(dir + "/" + dir_list[i])
        image = np.asarray(image, dtype = 'uint8')
        
        #image = adjust_dynamic_range(image, [0, PIXEL_MAX], [0, 1])
        #image = (image - 0.5)/0.5
    
        img_tensor[i] = image
        
    return(img_tensor)

def save_array(array, dir, dr_in, dr_out, N):
    
    for i in range(N):
        
        slice = array[i]
        slice = adjust_dynamic_range(slice, dr_in, dr_out)
        slice = np.rint(slice).clip(0, 255)
        slice = slice.astype('uint8')
        
        png.from_array(slice, mode = 'L' + ';8').save(dir + "/" + str(i).zfill(5) + ".png")
        
#%%

array = load_array(in_dir)

#%%

Ndata = array.shape[0]

sum = 0
for i in range(Ndata):
    img = array[i]
    if np.sum(img) == 0:
        sum +=1
        
ratio = sum/Ndata

print(ratio)
print(sum)
#save_array(array, out_dir, [0, 6], [0, 255], 500)
