'''
Reconstruction U-net for 3D mri data adjusted with ReLU activation function

https://arxiv.org/abs/1701.03056
'''
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, PReLU, Add, Concatenate, Input, Reshape, Dropout, Activation, Layer, Activation, UpSampling3D, AveragePooling3D
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal, Zeros, Ones
import numpy as np
import random
import tensorflow as tf
import math
from tfk_instance_norm import InstanceNormalization
from tensorflow.keras.layers import Lambda

############ Helpful functions ###########

def getShape(x):
    inputShape = []
    for i in range(1,5):
        inputShape.append(int(x.get_shape()[i]))
    return tuple(inputShape)

############ Contracting block ###########

def contrac_block(inp,features):
    conv = Conv3D(features, (1,1,1), strides=(2, 2,2), padding='valid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(inp)
    batch = InstanceNormalization(axis=4)(conv)
    Pre = PReLU()(batch)
    conv2 = Conv3D(features, (3,3,3), strides=(1, 1,1), padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(Pre)
    add = Add()([conv,conv2])
    batch2 = InstanceNormalization(axis=4)(add)
    Pre2 = PReLU()(batch2)
    
    return Pre2

############ Expanding block ###########

def expand_block(inp,inp2,features):
    conv = Conv3D(features, (1,1,1), strides=(1,1,1), padding='valid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(inp)
    batch = InstanceNormalization(axis=4)(conv)
    Pre = PReLU()(batch)

    # Deconvolution
    outShap = list(getShape(conv))
    for i in range(3):
        outShap[i]=2*outShap[i]
    outShap.insert(0,None)
    outShap = tuple(outShap)
    deconv = Conv3DTranspose(features, (1, 1, 1),strides=(2,2,2), padding='valid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(Pre)

    batch2 = InstanceNormalization(axis=4)(deconv)
    Pre2 = PReLU()(batch2)
    merg = Concatenate(axis=-1)([Pre2,inp2])

    #Convolution
    conv2 = Conv3D(features, (3,3,3), strides=(1,1,1), padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(merg)
    batch3 = InstanceNormalization(axis=4)(conv2)
    Pre3 = PReLU()(batch3)
    
    return Pre3

############ Reconstruction U-net architecture ###########

def RecUnet():
    inp = Input(shape=[128, 128, 128, 1])
    conv = Conv3D(8, (3,3,3), strides=(1,1,1), padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(inp)
    batch = InstanceNormalization(axis=4)(conv)
    Pre = PReLU()(batch)

    #Contracting Blocks
    con1 = contrac_block(batch,16)
    con2 = contrac_block(con1,32)
    con3 = contrac_block(con2,64)

    #Expanding Blocks
    exp1 = expand_block(con3,con2,32)
    exp2 = expand_block(exp1,con1,16)
    exp3 = expand_block(exp2,Pre,8)

    conv1 = Conv3D(3, (1,1,1), strides=(1,1,1), padding='valid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(exp3)

    conv2 = Conv3D(3, (1,1,1), strides=(1,1,1), padding='valid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(exp2)

    conv3 = Conv3D(3, (1,1,1), strides=(1,1,1), padding='valid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(exp1)
    up = UpSampling3D(size=(2, 2, 2))(conv3)

    merg = Add()([up,conv2])
    up2 = UpSampling3D(size=(2, 2, 2))(merg)

    merg2 = Add()([up2,conv1])

    predConv = Conv3D(1, (1,1,1), strides=(1,1,1), padding='valid',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(merg2)

    out = tf.keras.activations.relu(predConv)
    

    return Model(inputs=inp, outputs=out)
