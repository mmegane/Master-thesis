#%%
import os
import warnings

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

import keras.backend as K
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

# Allow growth of GPU memory (otherwise it will look like all the memory is being used, even if you only use 10 MB)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
#%%

from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

print("Training images have size {} and labels have size {} ".format(Xtrain.shape, Ytrain.shape))
print("Test images have size {} and labels have size {} \n ".format(Xtest.shape, Ytest.shape))

# Reduce the number of images for training and testing to 10000 and 2000 respectively, 
# to reduce processing time for this laboration
Xtrain = Xtrain[0:10000]
Ytrain = Ytrain[0:10000]

Xtest = Xtest[0:2000]
Ytest = Ytest[0:2000]

print("Reduced training images have size %s and labels have size %s " % (Xtrain.shape, Ytrain.shape))
print("Reduced test images have size %s and labels have size %s \n" % (Xtest.shape, Ytest.shape))

# Check that we have some training examples from each class
for i in range(10):
    print("Number of training examples for class {} is {}" .format(i,np.sum(Ytrain == i)))
    
#%%

plt.figure(figsize=(12,4))
for i in range(18):
    idx = np.random.randint(7500)
    label = Ytrain[idx,0]
    
    plt.subplot(3,6,i+1)
    plt.tight_layout()
    plt.imshow(Xtrain[idx])
    plt.title("Class: {} ({})".format(label, classes[label]))
    plt.axis('off')
plt.show()

#%%
from sklearn.model_selection import train_test_split

# Use train_test_split function to divide training data into training and validation
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.25)

print('The training, validation and testing set are made of {}, {} and {} images respectively.'.format(Xtrain.shape[0], Xval.shape[0], Xtest.shape[0]))

#%%

print("Data type is ",Xtrain.dtype)

# Convert datatype for Xtrain, Xval, Xtest, to float32
Xtrain = Xtrain.astype('float32')
Xval = Xval.astype('float32')
Xtest = Xtest.astype('float32')

Xtrain = Xtrain / 127.5 - 1
Xval = Xval / 127.5 - 1
Xtest = Xtest / 127.5 - 1

print("Data type is ",Xtrain.dtype)

#%%

from keras.utils import to_categorical

print("Shape of Ytrain before hot encoding is ", Ytrain.shape)

numClasses = 10

# Your code

print("Shape of Ytrain after hot encoding is ",Ytrain.shape)

#%%

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy as CC
from keras import regularizers

# Set seed from random number generator, for better comparisons
from numpy.random import seed
seed(123)

def build_CNN(input_shape, reg_parameter=0.0):

    # Setup a sequential model
    model = Sequential()

    # Add layers to the model
    
    # Your code
    
    # Compile model
    
    # Your code
    
    return model

#%%
    
# Setup some training parameters (feel free to change)
batch_size = 100
epochs = 100

# Build model
input_shape = Xtrain.shape[1:]
model = build_CNN(input_shape)

# Train the model
history = # Your code

# Evaluate the trained model on test set, not used in training or validation
score = model.evaluate(Xtest, Ytest, batch_size = batch_size, verbose=0)
print('Test loss: %.4f' % score[0])
print('Test accuracy: %.4f' % score[1])

# %%

# Plot training and validation losses and accuracy
val_loss, val_acc, loss, acc = history.history.values()

plt.figure(figsize=(10,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['Training','Validation'])

plt.figure(figsize=(10,4))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['Training','Validation'])

plt.show()

# %%