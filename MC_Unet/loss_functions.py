'''
A loss function used for evaluation of NN model performance in image segmentation 
'''

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

import tensorflow as tf
from tensorflow.keras import backend as K


def dice_score(y_true, y_pred, eps=1e-8):
    
    predShape = y_pred.get_shape().as_list()
    # change shape of predictions
    y_pred = tf.reshape(y_pred, (-1, predShape[-1]))
    y_true = tf.reshape(y_true, (-1, predShape[-1]))

    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps) # add a small value

    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=0, keepdims=True)
    denominator = tf.reduce_sum(y_true + y_pred, axis=0, keepdims=True)

    return tf.reduce_sum((numerator + 1) / (denominator + 1) / predShape[-1])

def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)



