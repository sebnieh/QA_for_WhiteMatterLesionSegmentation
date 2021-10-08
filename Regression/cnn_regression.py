'''
Architecture of CNN regression
'''

from tensorflow.keras.layers import Conv3D, PReLU, Input, Dropout, Activation, Layer, AveragePooling3D, LeakyReLU, BatchNormalization, MaxPooling3D, GlobalAveragePooling3D, Dense, Flatten
from tensorflow.keras.models import Model


def create_cnn(input_shape = (128, 128, 128, 3), filters=(16, 32, 64, 128)): 
    inputs = Input(shape=input_shape) 
    for (i, f) in enumerate(filters): 
        if i == 0:
            x = inputs
        x = Conv3D(f, (3, 3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(1, activation="linear")(x) 
    model = Model(inputs, x)
    return model
