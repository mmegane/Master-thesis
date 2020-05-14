#%%
import os
import warnings

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

import keras.backend as K
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

# Allow growth of GPU memory (otherwise it will look like all the memory is being used, even if you only use 10 MB)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

#%%

import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.utils import to_categorical

classes = ["Background", "Retinal Vessels"]
Nclasses = len(classes)
newSize = 512

image_list = []
label_list = []
for i in range(30):
    filename_im = '/nobackup/local/mehfo331/Thesis/Misc/input/isbi2015/train/image/' +str(i)+ '.png'
    filename_gt = '/nobackup/local/mehfo331/Thesis/Misc/input/isbi2015/train/label/' +str(i)+ '.png'
    im = img_to_array(load_img(filename_im,target_size=(newSize,newSize)))/255
    image_list.append(im)
    label = img_to_array(load_img(filename_gt,target_size=(newSize,newSize),color_mode="grayscale"))/255
    label_list.append(label)
    
# Convert lists into numpy arrays    
imds = np.asarray(image_list)
gtds = np.asarray(label_list, dtype=int)

# Show the first retinal image and the first ground truth image
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(array_to_img(imds[0]))
plt.title('Original Image')
plt.subplot(122)
plt.imshow(array_to_img(gtds[0,:,:]))
plt.title('Ground-truth')

# Transform ground truth images into categorical
gtds = to_categorical(gtds, Nclasses)
print('The image dataset has shape: {}'.format(imds.shape))
print('The ground-truth dataset has shape: {}'.format(gtds.shape))

# %%

from sklearn.model_selection import train_test_split

X, Xtest, Y, Ytest = train_test_split(imds, gtds, test_size=0.1)
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X, Y, test_size=0.15)

print('The training, validation and testing set are made by {}, {} and {} images respectively.'.format(Xtrain.shape[0], Xvalid.shape[0], Xtest.shape[0]))
print('\nThe training images dataset has shape: {}'.format(Xtrain.shape))
print('The training ground-truth dataset has shape: {}'.format(Ytrain.shape))
print('The validation images dataset has shape: {}'.format(Xvalid.shape))
print('The validation ground-truth dataset has shape: {}'.format(Yvalid.shape))
print('The testing images dataset has shape: {}'.format(Xtest.shape))
print('The testing ground-truth dataset has shape: {}'.format(Ytest.shape))

# %%

def myPreProc(imgs, mean):  
    X = imgs - mean
    return X

# Do image pre-processing for the training, validation and testing set separately!
meanTrain = np.mean(Xtrain, axis=(0,1,2))
Xtrain_preprocessed = myPreProc(Xtrain, meanTrain)
Xvalid_preprocessed = myPreProc(Xvalid, meanTrain)
Xtest_preprocessed = myPreProc(Xtest, meanTrain)

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(array_to_img(Xtrain[0,:,:,:]))
plt.title('Original Image')
plt.subplot(122)
plt.imshow(array_to_img(Xtrain_preprocessed[0,:,:,:]))
plt.title('Pre-processed image')

# %%
from keras.optimizers import Adam, SGD
from keras_unet.models import vanilla_unet, custom_unet
from keras_unet.metrics import iou
from keras_unet.losses import jaccard_distance

model = custom_unet(input_shape=(512, 512, 3),
                    use_batch_norm=True,
                    num_classes = 2,
                    output_activation = 'relu')

model.compile(loss = 'binary_crossentropy',
              #loss=jaccard_distance,
              optimizer=Adam(lr=1e-4),
              metrics=[iou])

# %%

from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 1

callbacks = [ModelCheckpoint('/nobackup/local/mehfo331/Thesis/Misc/unet-weights',
                             verbose = 1,save_best_only = True, save_weights_only = True),
                             EarlyStopping(patience = 10)]

results = model.fit(Xtrain_preprocessed, Ytrain, validation_data= (Xvalid_preprocessed, Yvalid), batch_size = batch_size, epochs= 500, callbacks = callbacks)

# %%

#model.load_weights('/nobackup/local/mehfo331/Thesis/Misc/unet-weights')

from keras_unet.utils import plot_segm_history

plot_segm_history(results)
# %%

model.load_weights('/nobackup/local/mehfo331/Thesis/Misc/unet-weights')
loss, metric = model.evaluate(Xtest_preprocessed, Ytest, batch_size = batch_size)
y_pred = model.predict(Xtest_preprocessed)
print(loss)
print(metric)

#%%

from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=Xtest, mask_imgs=Ytest, pred_imgs=y_pred, nm_img_to_plot=9)