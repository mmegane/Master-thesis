import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import os

#%%

mask_path = "/nobackup/data/mehfo331/Data/Slices/z/Padded/Masks_complete/GAN"
img_path = "/nobackup/data/mehfo331/Data/Slices/z/Padded/t1ce/GAN"

img_mask = Image.open(mask_path + "/09999.png")
mask_array = np.asarray(img_mask, dtype = 'uint8')

img = Image.open(img_path + "/09999.png")
img_array = np.asarray(img, dtype = 'uint16')

from matplotlib import pyplot as plt

plt.figure(figsize = (12,12))
plt.subplot(131)
plt.imshow(mask_array, cmap = "gray")
plt.title('mask')
plt.subplot(132)
plt.imshow(img_array, cmap = "gray")
# plt.title('MR image')

#%%

path = "/nobackup/data/mehfo331/Data/Slices/z/Padded/Masks_complete/GAN"

dir = os.listdir(path)
Ndata = len(dir)

img_tensor = np.zeros((Ndata, 256, 256), dtype = 'uint8')

for i in range(Ndata):
    img = Image.open(path + "/" + dir[i])
    img_array = np.asarray(img, dtype = 'uint8')
    
    img_tensor[i] = img_array
    

#%%


path = "/nobackup/data/mehfo331/Data/Slices/z/Padded/t1ce/Training"
img = Image.open(path + "/00000.png")
img = img.convert(mode = 'RGB', colors = 65504)
img_array = np.asarray(img, dtype = 'float32')

#%%

from matplotlib import pyplot as plt

plt.subplot()
plt.imshow(img_array)

#%%


path = "/nobackup/data/mehfo331/Code/SPADE/checkpoints/PGAN-brains/web/images"
img = Image.open(path + "/epoch001_iter2200_real_image.png")
img_array = img_to_array(img, dtype = 'float16')



#%%


path = "/nobackup/data/mehfo331/Code/progressive_growing_of_gans/results/043-fake-images-41"
#path = "/nobackup/data/mehfo331/Data/Slices/z/Padded/Masks_complete/Training"

img = Image.open(path + "/041-pgan-masks_complete_no_blacks-preset-v2-1gpu-fp32-network-snapshot-008800-000066.png")
#img = Image.open(path + "/train_mask00007.png")
img_array = img_to_array(img, dtype = 'uint8')

print(np.unique(img_array, return_counts = True))

img_array = img_array.copy()
#img_array[img_array == 0] = 0
#img_array[img_array == 1] = 0
#img_array[img_array == 2] = 0
#img_array[img_array == 3] = 0
#img_array[img_array == 4] = 0
#img_array[img_array == 5] = 0
#img_array[img_array == 6] = 0

from matplotlib import pyplot as plt

plt.subplot()
plt.imshow(img_array[:,:,0], cmap = "gray")


# 3 = WM, 2 = GM, 1 = CSF

# 0 = BG, 1 = CSF, 2 = GM, 3 = WM, 4 = NCR/NET, 5 = ED, 6 = ET

#%%

path ="/nobackup/data/mehfo331/Data/Slices/z/Padded/Masks_complete/Training"



img = Image.open(path + "/train_mask00014.png")
img_array = np.asarray(img, dtype = 'uint8')
img_array = img_array[np.newaxis, :, :]

print(np.unique(img_array, return_counts = True))

# img_array = img_array.astype(np.float32)
# img_array = (img_array[:, 0::2, 0::2] + img_array[:, 0::2, 1::2] + img_array[:, 1::2, 0::2] + img_array[:, 1::2, 1::2]) * 0.25
quant = np.rint(img_array).clip(0, 255).astype(np.uint8)

print(np.unique(quant, return_counts = True))

#%%

#path ="/nobackup/data/mehfo331/Data/Slices/z/Padded/Masks_complete/Training"
path = "/nobackup/data/mehfo331/Data/Slices/z/Padded/t1ce/Training"


img = Image.open(path + "/train_image00014.png")
img_array = np.asarray(img, dtype = 'float16')
img_array = img_array[np.newaxis, :, :]

img_array = img_array.astype(np.float32)
img_array = (img_array[:, 0::2, 0::2] + img_array[:, 0::2, 1::2] + img_array[:, 1::2, 0::2] + img_array[:, 1::2, 1::2]) * 0.25
quant = np.rint(img_array).clip(0, 65535).astype(np.float16)


#%%

from matplotlib import pyplot as plt

#img_array = np.rint(img_array/42)
#img_array = np.rint(img_array)
#print(np.unique(img_array, return_counts = True))

plt.subplot()
plt.imshow(img_array[:,:,0], cmap = "gray")

