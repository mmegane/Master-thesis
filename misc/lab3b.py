# %%

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

# %%

import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.utils import to_categorical

classes = ["Background", "Retinal Vessels"]
Nclasses = len(classes)
newSize = 320

image_list = []
label_list = []
for i in range(60):
    filename_im = 'Data/Images/image' +str(i+1)+ '.tif'
    filename_gt = 'Data/Masks/mask' +str(i+1)+ '.gif'
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

from sklearn.utils.class_weight import compute_class_weight

y = Ytrain[:,:,:,1].flatten()
class_weights = compute_class_weight('balanced', np.arange(Nclasses), y)
print('The class weights that belong to the background and to the foreground are respectively:\n{}'.format(class_weights))

# %%

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

class unet(object):
    
    def __init__(self, img_size, Nclasses, class_weights, model_name='myWeights.h5', Nfilter_start=64, depth=3, batch_size=3):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        self.model_name = model_name
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size

        self.model = Sequential()
        inputs = Input(img_size)
    
        def dice(y_true, y_pred, eps=1e-5):
            num = 2.*K.sum(self.class_weights*K.sum(y_true * y_pred, axis=[0,1,2]))
            den = K.sum(self.class_weights*K.sum(y_true + y_pred, axis=[0,1,2]))+eps
            return num/den

        def diceLoss(y_true, y_pred):
            return 1-dice(y_true, y_pred)       
    
        def bceLoss(y_true, y_pred):
            bce = K.sum(-self.class_weights*K.sum(y_true*K.log(y_pred), axis=[0,1,2]))
            return bce
        
        # This is a help function that performs 2 convolutions (filter size (3 x 3), he normal initialization, same padding),
        # each followed by batch normalization and ReLu activations, Nf is the number of filters
        def convs(layer, Nf):
            
            # Your code
            
            conv_1 = Conv2D(filters = Nf, kernel_size = (3,3), padding = 'same', kernel_initializer = 'he_normal')
            x = conv_1(layer)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
             
            conv_2 = Conv2D(filters = Nf, kernel_size = (3,3), padding = 'same', kernel_initializer = 'he_normal')
            x = conv_2(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            return x
        
        # This is a help function that defines what happens in each layer of the encoder (downstream),
        # which calls "convs" and then Maxpooling (2 x 2). Save each layer (before max pooling) 
        # for later concatenation in the upstream.
        def encoder_step(layer, Nf):
            
            # Your code

            y = convs(layer, Nf)
            x = MaxPooling2D(pool_size = (2,2))(y)
            
            return y, x
            
        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains transpose convolution (filter size (3 x 3), stride (2,2), he normal initialization, same padding)
        # batch normalization, concatenation with corresponding layer from encoder, and lastly "convs"
        def decoder_step(layer, layer_to_concatenate, Nf):
            
            # Your code
            
            transposeConv = Conv2DTranspose(filters = Nf, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer  = 'he_normal')
            
            x = transposeConv(layer)
            x = BatchNormalization()(x)
            x = concatenate([layer_to_concatenate, x])
            x = convs(x, Nf)
            
            return x
        
        layers_to_concatenate = []
        x = inputs
        
        # Make encoder with 'self.depth' layers, 
        # note that the number of filters in each layer will double compared to the previous "step" in the encoder
        for d in range(self.depth-1):
            y,x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(y)
            
        # Make bridge, that connects encoder and decoder using "convs" between them. 
        # Use Dropout before and after the bridge, for regularization. Use dropout probability of 0.2.
        x = Dropout(0.2)(x)
        x = convs(x,self.Nfilter_start*np.power(2,self.depth-1))
        x = Dropout(0.2)(x)        
        
        # Make decoder with 'self.depth' layers, 
        # note that the number of filters in each layer will be halved compared to the previous "step" in the decoder
        for d in range(self.depth-2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start*np.power(2,d))            
            
        # Make classification (segmentation) of each pixel, using convolution with 1 x 1 filter
        final = Conv2D(filters=self.Nclasses, kernel_size=(1,1), activation = 'softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(loss=diceLoss, optimizer=Adam(lr=1e-4), metrics=['accuracy',dice]) 
        
    def train(self, X, Y, x, y, nEpochs):
        print('Training process:')
        callbacks = [ModelCheckpoint(self.model_name, verbose=1, save_best_only=True, save_weights_only=True),
                    EarlyStopping(patience=10)]
        results = self.model.fit(X, Y, validation_data=(x,y), batch_size=self.batch_size, epochs=nEpochs, callbacks=callbacks)
        return results
        
    def train_with_aug(self, im_gen_train, gt_gen_train, im_gen_valid, gt_gen_valid, nEpochs):       
        print('Training process:')
        # we save in a dictionary the metrics obtained after each epoch
        results_dict = {}
        results_dict['loss'] = []
        results_dict['acc'] = []
        results_dict['dice'] = []
        results_dict['val_loss'] = []
        results_dict['val_acc'] = []
        
        val_loss0 = np.inf
        steps_val_not_improved = 0
        for e in range(nEpochs):
            print('\nEpoch {}/{}'.format(e+1, nEpochs))
            Xb_train, Yb_train = im_gen_train.next(), gt_gen_train.next()
            Xb_valid, Yb_valid = im_gen_valid.next(), gt_gen_valid.next()
            # Transform ground truth images into categorical
            Yb_train = to_categorical(np.argmax(Yb_train, axis=-1), self.Nclasses)
            Yb_valid = to_categorical(np.argmax(Yb_valid, axis=-1), self.Nclasses)               

            results = self.model.fit(Xb_train, Yb_train, validation_data=(Xb_valid,Yb_valid), batch_size=self.batch_size)

            if results.history['val_loss'][0] <= val_loss0:
                self.model.save_weights(self.model_name)
                print('val_loss decreased from {:.4f} to {:.4f}. Hence, new weights are now saved in {}.'.format(val_loss0, results.history['val_loss'][0], self.model_name))
                val_loss0 = results.history['val_loss'][0]
                steps_val_not_improved = 0
            else:
                print('val_loss did not improved.')
                steps_val_not_improved += 1

            # saving the metrics
            results_dict['loss'].append(results.history['loss'][0])
            results_dict['acc'].append(results.history['acc'][0])
            results_dict['dice'].append(results.history['dice'][0])
            results_dict['val_loss'].append(results.history['val_loss'][0])
            results_dict['val_acc'].append(results.history['val_acc'][0])
            results_dict['val_dice'].append(results.history['val_dice'][0])
            
            if steps_val_not_improved==10:
                print('\nThe training stopped because the network after 10 epochs did not decrease it''s validation loss.')
                break

        return results_dict
    
    def evaluate(self, X, Y):
        print('Evaluation process:')
        score, acc, dice = self.model.evaluate(X,Y,self.batch_size)
        print('Accuracy: {:.4f}'.format(acc*100))
        print('Dice: {:.4f}'.format(dice*100))
        return acc, dice
    
    def predict(self, X):
        print('Segmenting unseen image')
        segmentation = self.model.predict(X)
        return segmentation
    
# %%

img_size = Xtrain_preprocessed[0].shape
net = unet(img_size, Nclasses, class_weights, Nfilter_start=64, batch_size=3, depth=5)
# net.model.summary()
results = net.train(Xtrain_preprocessed, Ytrain, Xvalid_preprocessed, Yvalid, nEpochs=50)
net.model.load_weights('myWeights.h5')
print(' ')
acc, dice = net.evaluate(Xtest_preprocessed,Ytest)

# accuracy trend
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
# dice trend
plt.plot(results.history['dice'])
plt.plot(results.history['val_dice'])
plt.ylabel('Dice')
plt.xlabel('Epochs')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
# loss trend
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# %%

Ypred = net.predict(Xtest_preprocessed)

plt.figure(figsize=(12,12))
plt.subplot(131)
plt.imshow(Xtest[0])
plt.title('Image')
plt.subplot(132)
# Segmentation in each pixel is the class of the strongest prediction / activation
plt.imshow(np.argmax(Ypred[0], axis=-1))
plt.title('Segmentation result')
plt.subplot(133)
plt.imshow(Ytest[0,:,:,1])
plt.title('Ground truth segmentation')

# %%
from keras.preprocessing.image import ImageDataGenerator

def apply_augmentation(X, Y, N_new_images):
    data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip = True,
                     vertical_flip = True,
                     zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = np.random.randint(123456789)
    image_generator = image_datagen.flow(X, batch_size=N_new_images, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=N_new_images, seed=seed)
    
    return image_generator, mask_generator

image_generator_train, mask_generator_train = apply_augmentation(Xtrain_preprocessed, Ytrain, len(Xtrain))
image_generator_valid, mask_generator_valid = apply_augmentation(Xvalid_preprocessed, Yvalid, len(Xvalid))

# let's generate one batch of augmented images for the training set and plot few of them
X_batch, Y_batch = image_generator_train.next(), mask_generator_train.next()

plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(array_to_img(X_batch[i]))
    plt.subplot(2,5,i+6)
    plt.imshow(np.argmax(Y_batch[i], axis=-1))
# %%

img_size = Xtrain_preprocessed[0].shape
modelName = 'myWeightsAug.h5'

net = unet(img_size, Nclasses, class_weights, modelName, Nfilter_start=64, batch_size=3, depth=5)
#net.model.summary()
results = net.train_with_aug(image_generator_train, mask_generator_train, image_generator_valid, mask_generator_valid, nEpochs=50)
net.model.load_weights(modelName)
print(' ')
acc, dice = net.evaluate(Xtest_preprocessed,Ytest)

# accuracy trend
plt.plot(results['acc'])
plt.plot(results['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
# dice trend
plt.plot(results['dice'])
plt.plot(results['val_dice'])
plt.ylabel('Dice')
plt.xlabel('Epochs')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
# loss trend
plt.plot(results['loss'])
plt.plot(results['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# %%

Ypred = net.predict(Xtest_preprocessed)

plt.figure(figsize=(12,12))
plt.subplot(131)
plt.imshow(Xtest[0])
plt.title('Image')
plt.subplot(132)
plt.imshow(np.argmax(Ypred[0], axis=-1))
plt.title('Segmentation result')
plt.subplot(133)
plt.imshow(Ytest[0,:,:,1])
plt.title('Ground truth segmentation')