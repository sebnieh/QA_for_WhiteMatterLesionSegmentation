'''
Training script for folds 1-5.
'''
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import rec
import numpy as np
import random
import pickle
import os
import gener
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # "0, 1" for multiple

data_path = '/opt/Lesion_Segmentation/reg_net/'


def rescale(data, max=255):
    '''
    rescale array to [0 max]
    '''

    data = data-np.min(np.ravel(data))
    data = max* (data/np.max(np.ravel(data)))
    return data

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

for i in np.arange(1,6):

    print("Loading the data...")
    ######################## LOAD VAL X DATA #########################
    with open(data_path+str(i)+'x_val.pkl', 'rb') as f:
        x = pickle.load(f)
    x = x[:,0,:,:,:,1]    
    with open(data_path+str(i)+'gt_val.pkl', 'rb') as f:
        gt = pickle.load(f)
    gt = gt[:,0,:,:,:,1]
    x[gt>0] = 0
    x = rescale(x, max=1)
    x_val = np.expand_dims(x, axis = 4)
    ######################## LOAD VAL Y DATA #########################
    with open(data_path+str(i)+'x_val.pkl', 'rb') as f:
        y = pickle.load(f)
    y = y[:,0,:,:,:,1]
    y = rescale(y, max=1)
    y_val = np.expand_dims(y, axis = 4)
    ######################## LOAD TRAIN X DATA #########################
    with open(data_path+str(i)+'x_train.pkl', 'rb') as f:
        x_tr = pickle.load(f)
    x_tr = x_tr[:,0,:,:,:,1]
    with open(data_path+str(i)+'gt_train.pkl', 'rb') as f:
        gt_tr = pickle.load(f)
    gt_tr = gt_tr[:,0,:,:,:,1]
    x_tr[gt_tr>0] = 0
    x_tr  = rescale(x_tr, max=1)
    x_tr = np.expand_dims(x_tr, axis = 4)
    ######################## LOAD TRAIN Y DATA #########################
    with open(data_path+str(i)+'x_train.pkl', 'rb') as f:
        y_tr = pickle.load(f)
    y_tr = y_tr[:,0,:,:,:,1]
    y_tr = rescale(y_tr, max=1)
    y_tr = np.expand_dims(y_tr, axis = 4)

    train = [x_tr, y_tr]
    val = [x_val, y_val]
    ######################## SET UP THE GENERATOR ########################
    genTra = gener.DataGenerator(list(range(len(x_tr))), train, batch_size=2, shuffle=True)
    genVal = gener.DataGenerator(list(range(len(x_val))), val, batch_size=2, shuffle=True)

    model = rec.RecUnet()
    model.compile(loss=ssim_loss, optimizer='adam', metrics = ['mae'])
    print('Initiating training ...')

    checkpoint = [CSVLogger('training_hist/'+str(i)+'training.csv', append=True, separator=','), # Save best model and history
                  ModelCheckpoint('models/'+str(i)+'best_model.hdf5', 
                                  monitor='val_loss', verbose=1, save_best_only= True, mode='min')]
    model.fit_generator(generator = genTra, epochs=1700,steps_per_epoch=34, validation_data=genVal, validation_steps=8,callbacks=checkpoint,use_multiprocessing=True)


