import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
#from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

class Unet(object):
    
    def __init__(self, img_size, Nclasses, class_weights, Nfilter_start = 64, depth = 5):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        #self.weights_path = weights_path
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        #self.batch_size = batch_size

        self.model = Sequential()
        inputs = Input(img_size)
 
        def weighted_dice_loss(y_true, y_pred):
            eps = 1e-5
                
            num = 2. * K.sum(self.class_weights * K.sum(y_true * y_pred, axis = [0,1,2]))
            den = K.sum(self.class_weights * K.sum(y_true + y_pred, axis = [0,1,2])) + eps
            
            loss = 1 - num/den
            return(loss)
        
        def dice_acc(y_true, y_pred):
            return (1 - weighted_dice_loss(y_true, y_pred))
        
        
        # This is a help function that performs 2 convolutions (filter size (3 x 3), he normal initialization, same padding),
        # each followed by batch normalization and ReLu activations, Nf is the number of filters
        
        def convs(layer, Nf):
            
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
        
            y = convs(layer, Nf)
            x = MaxPooling2D(pool_size = (2,2))(y)
            
            return y, x
            
        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains transpose convolution (filter size (3 x 3), stride (2,2), he normal initialization, same padding)
        # batch normalization, concatenation with corresponding layer from encoder, and lastly "convs"
        
        def decoder_step(layer, layer_to_concatenate, Nf):
            
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
            y,x = encoder_step(x, self.Nfilter_start * np.power(2,d))
            layers_to_concatenate.append(y)
            
        # Make bridge, that connects encoder and decoder using "convs" between them. 
        # Use Dropout before and after the bridge, for regularization. Use dropout probability of 0.2.
            
        #x = Dropout(0.2)(x)
        x = convs(x, self.Nfilter_start * np.power(2, self.depth - 1))
        #x = Dropout(0.2)(x)        
        
        # Make decoder with 'self.depth' layers, 
        # note that the number of filters in each layer will be halved compared to the previous "step" in the decoder
        
        for d in range(self.depth - 2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start * np.power(2,d))            
            
        # Make classification (segmentation) of each pixel, using convolution with 1 x 1 filter
            
        final = Conv2D(filters = self.Nclasses, kernel_size=(1,1), activation = 'softmax')(x)
        
        # Create model
        
        self.model = Model(inputs = inputs, outputs = final)
        
        self.model.compile(loss = weighted_dice_loss,
                           optimizer = Adam(lr = 1e-4),
                           metrics = [dice_acc]) 
    