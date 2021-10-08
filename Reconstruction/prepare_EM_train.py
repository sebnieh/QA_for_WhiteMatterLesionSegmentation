'''
This script is used to save error maps for train instances across 5-folds
'''

from tensorflow.keras.models import load_model
from tfk_instance_norm import InstanceNormalization
import utils_reg as ut
import numpy as np
import random
import pickle
import os 
import tensorflow as tf
import scipy.stats
def rescale(data, max=255):
    '''
    rescale array to [0 max]
    '''
    
    data = data-np.min(np.ravel(data))
    data = max* (data/np.max(np.ravel(data)))
    return data

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # "0, 1" for multiple

data_path = '/opt/Lesion_Segmentation/reg_net/'

for i in np.arange(5,6):
    print('FOLD:', i)
    print("Loading the train data...")
    ######################## LOAD train X DATA #########################
    with open(data_path+str(i)+'x_train.pkl', 'rb') as f:
        x = pickle.load(f)
    x = x[:,0,:,:,:,1]    
    with open(data_path+str(i)+'gt_train.pkl', 'rb') as f:
        gt = pickle.load(f)
    gt = gt[:,0,:,:,:,1]
    x[gt>0] = 0
    x = rescale(x, max=1)
    x_train = np.expand_dims(x, axis = 4)
    ######################## LOAD train Y DATA #########################
    with open(data_path+str(i)+'x_train.pkl', 'rb') as f:
        y = pickle.load(f)
    y = y[:,0,:,:,:,1]
    y = rescale(y, max=1)
    y_train = np.expand_dims(y, axis = 4)
    print(y_train.shape)
    model = load_model('models/'+str(i)+'best_model.hdf5',  
                   custom_objects={'ssim_loss':ssim_loss, 'InstanceNormalization':InstanceNormalization})
    
    predictions = model.predict(x_train[0:25])    
    em = predictions - y_train[0:25]
    
    predictions2 = model.predict(x_train[25:50])
    em2 = predictions2 - y_train[25:50]

    predictions3 = model.predict(x_train[50:68])
    em3 = predictions3 - y_train[50:68]
    em = np.concatenate((em, em2, em3), axis=0)
    # SAVE ERROR MAPS
    with open("/opt/"+str(i)+'em_train.pkl','wb') as f:
        pickle.dump(em, f)

