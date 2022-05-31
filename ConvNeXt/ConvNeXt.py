import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model,layers,Sequential
import tensorflow.keras.activations
import numpy as np
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import math
from keras_flops import get_flops
import tensorflow_addons as tfa

#################### ConvNeXt configs ####################
'''
ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3) 

ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)

ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)

ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)

ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)
'''
#-------------------------------------------------#
#              ConvNeXt Residual Block
#-------------------------------------------------#
def convnext_block(input, dim, drop_path=0.0):
    """
    ConvNeXt Block

    Input Parameters:

    input: input tensor
    dim: Number of filters/channels
    drop_path: Set to 0 by default
    ----------------------

    About the block:
      - 7x7 depthwise conv(can be implemented from Conv2D, no of groups = no of filters)
      - 2 1x1 convs or pointwise layers are equivalent to dense layers(or fully connected layer) operating independently on each channel
      - The number of channels in the first 1x1 conv is 4 times the input dimension.
      - The channels for the last 1x1 conv are similar to input channels. See the block.
      - Layer normalization layer comes before the first 1x1 conv
      - GELU layer is inserted between two 1x1 convs
      - There is a direct shortcut from the input of the block to the output
    """
    shortcurt = input  # shortcut connection
    x = layers.Conv2D(filters=dim, kernel_size=7, padding='same', groups=dim)(input)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(4 * dim)(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dense(dim)(x)

    # Stochastic depth
    if drop_path > 0:

        outputs = tfa.layers.StochasticDepth(1 - drop_path)([shortcurt, x])

    else:
        outputs = layers.add([x, shortcurt])

    return outputs


#-------------------------------------------------#
#             Stem block
#-------------------------------------------------#
'''
The stem of the network is made of 4x4 convolution layer with 
stride 4 followed by a layer normalization layer. 
The stem network comes at the beginning of ConvNeXt.
'''
def stem(input, dim):

  x = layers.Conv2D(filters=dim, kernel_size=4, strides=4)(input)
  x = layers.LayerNormalization(epsilon=1e-6)(x)

  return x


#-------------------------------------------------#
#             Downsampling Block
#-------------------------------------------------#
'''
Downsampling layers are layer normalization layer 
and 2x2 convolution layer. Downsampling layers are 
used for reducing the spatial resolution and are 
inserted between ConvNeXt stages.
因为直接使用down_sampling layer使训练不稳定，因此在每个
下采样层前面增加了LN来稳定训练

'''
def downsampling_layers(input, dim):

  x = layers.LayerNormalization(epsilon=1e-6)(input)
  x = layers.Conv2D(filters=dim, kernel_size=2, strides=2)(x)

  return x


#-------------------------------------------------#
#        Building the Whole ConvNeXt Network
#-------------------------------------------------#
def convnext_model(input_shape=(224, 224, 3),
                   stage=[3,3,9,3],
                   dims=[96, 192, 384, 768],
                   num_classes=1000,
                   name=None):
    '''
    ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)

    ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)

    ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)

    ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)

    ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)
    '''

    input = layers.Input(input_shape)

    # stem
    x = stem(input, dims[0])

    # Convnext stage 1 x3, dim[0] = 96,128,192,256
    for _ in range(stage[0]):
        x = convnext_block(x, dims[0])

    # Downsampling layers + stage 2 x3, dim[1] = 192,256,384,512
    x = downsampling_layers(x, dims[1])
    for _ in range(stage[1]):
        x = convnext_block(x, dims[1])

    # Downsampling layers + stage 3 x9, dim[2] = 384,512,768,1024
    x = downsampling_layers(x, dims[2])
    for _ in range(stage[2]):
        x = convnext_block(x, dims[2])

    # Downsampling layers + stage 4 x3, dim[3] = 768,1024,1536,2048
    x = downsampling_layers(x, dims[3])
    for _ in range(stage[3]):
        x = convnext_block(x, dims[3])

    # Classification head: Global average pool + layer norm + fully connected layer
    x = layers.GlobalAvgPool2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    output = layers.Dense(units=num_classes, activation='softmax')(x)

    model = Model(input, output, name='ConvNeXt'+name)

    return model

def ConvNeXt_Tiny(input_shape=(224, 224, 3),
                  stage=[3,3,9,3],
                  dims=[96, 192, 384, 768],
                  num_classes=1000,
                  name='-T'):
    return convnext_model(input_shape=input_shape,stage=stage,dims=dims,num_classes=num_classes,name=name)

def ConvNeXt_Small(input_shape=(224, 224, 3),
                   stage=[3,3,27,3],
                   dims=[96, 192, 384, 768],
                   num_classes=1000,
                   name='-S'):
    return convnext_model(input_shape=input_shape, stage=stage, dims=dims, num_classes=num_classes,name=name)

def ConvNeXt_Base(input_shape=(224, 224, 3),
                  stage=[3,3,27,3],
                  dims=[128, 256, 512, 1024],
                  num_classes=1000,
                  name='-B'):
    return convnext_model(input_shape=input_shape, stage=stage, dims=dims, num_classes=num_classes,name=name)

def ConvNeXt_Large(input_shape=(224, 224, 3),
                   stage=[3,3,27,3],
                   dims=[192, 384, 768, 1536],
                   num_classes=1000,
                   name='-L'):
    return convnext_model(input_shape=input_shape, stage=stage, dims=dims, num_classes=num_classes,name=name)

def ConvNeXt_XL(input_shape=(224, 224, 3),
                stage=[3,3,27,3],
                dims=[256, 512, 1024, 2048],
                num_classes=1000,
                name='-XL'):
    return convnext_model(input_shape=input_shape, stage=stage, dims=dims, num_classes=num_classes,name=name)

model = ConvNeXt_Large()
model.summary()

plot_model(model,to_file='convnext.png',show_layer_names=True,show_shapes=True,dpi=128)