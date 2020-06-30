import numpy as np
from PIL import Image
import png
import os

#image_shape = (256,256)

train_dir = "/nobackup/data/mehfo331/Data/Slices/z/New/Masks_complete/Training/Fifth"
gan_dir = "/nobackup/data/mehfo331/Data/Slices/z/New/Masks_complete/GAN/Fifth/Raw"

#%%

def load_array(dir):
    dir_list = sorted(os.listdir(dir))
    dir_list = [s for s in dir_list if '.png' in s]
    
    Ndata = len(dir_list)

    img_tensor = np.zeros((Ndata, 256,256), dtype = 'uint8')

    for i in range(Ndata):
        mask = Image.open(dir + "/" + dir_list[i])
        mask_array = np.asarray(mask, dtype = 'uint8')
    
        img_tensor[i] = mask_array
        
    return(img_tensor)
    
#%%

# Load training data:

train_tensor = load_array(train_dir)
    
#%%
axis = 0
#axis = None

train_mean = np.mean(train_tensor, axis = axis)
train_std = np.std(train_tensor, axis = axis)

#%%

# Load GAN data:

dir = gan_dir
gan_tensor = load_array(gan_dir)

gan_tensor = gan_tensor[50000:100000]

#%%

gan_mean = np.mean(gan_tensor, axis = axis)
gan_std = np.std(gan_tensor, axis = axis)

#%%

ord = 2

train_Z = (train_tensor - train_mean)/(train_std + 1e-5)
train_Z = np.reshape(train_Z, (train_Z.shape[0], 256*256))
train_Z_L = np.linalg.norm(train_Z, ord = ord, axis = 1)
#train_Z_L = np.linalg.norm(train_Z, ord = ord, axis = (1,2))

gan_Z = (gan_tensor - train_mean)/(train_std + 1e-5)
gan_Z = np.reshape(gan_Z, (gan_Z.shape[0], 256*256))
gan_Z_L = np.linalg.norm(gan_Z, ord = ord, axis = 1)
#gan_Z_L = np.linalg.norm(gan_Z, ord = ord, axis = (1,2))
#%%

import matplotlib.pyplot as plt

x = np.arange(200, 1000, 0.1)
y_train = np.zeros(len(x))
y_gan = np.zeros(len(x))

for i in range(len(x)):
    threshold = x[i]
    
    train_indeces = np.where(train_Z_L < threshold)
    gan_indeces = np.where(gan_Z_L < threshold)
     
    ratio_train = np.size(train_indeces)/train_tensor.shape[0]
    ratio_gan = np.size(gan_indeces)/gan_tensor.shape[0]
    
    y_train[i] = ratio_train
    y_gan[i] = ratio_gan

# train = blue
# gan = orange

plt.plot(x, y_train)
plt.plot(x, y_gan)
plt.show()
#%%

#Candidates (axis = 0, ord = 2): 410, 490, 100

threshold =  500
train_indeces = np.where(train_Z_L < threshold)[0]
gan_indeces = np.where(gan_Z_L < threshold)[0]

ratio_train = np.size(train_indeces)/train_tensor.shape[0]
ratio_gan = np.size(gan_indeces)/gan_tensor.shape[0]

print(ratio_train)
print(ratio_gan)

#%%
def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

path = "/nobackup/data/mehfo331/Data/Slices/z/New/Masks_complete/GAN/Fifth/Preprocessed/Kept"

# files = os.listdir(path)
# for i in range(len(files)):
#     os.remove(path + "/" + files[i])

for i in range(len(gan_indeces)):

    index = gan_indeces[i] 
    slice = gan_tensor[index]
    
    #slice = adjust_dynamic_range(slice, [0,6], [0, 255])
    
    slice = slice.astype('uint8')
    
    png.from_array(slice, mode = 'L' + ';8').save(path + "/" + str(49073 + i).zfill(5) + ".png")