'''
Training script for folds 1-5 with an input of image-prediction maps pair.
'''
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import utils as ut
import cnn_regression as net
import numpy as np
import random
import pickle
import os 
import gener
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple

input_shape = (128, 128, 128, 2)
data_path = '/opt/Lesion_Segmentation/reg_net/'


for i in np.arange(1,6):

    print("Loading the data...")
    ######################### LOAD train DATA ######################
    with open(data_path+str(i)+'x_train.pkl', 'rb') as f:
        x = pickle.load(f)
    with open(data_path+str(i)+'prediction_tr.pkl', 'rb') as f:
        prediction = pickle.load(f)
    dice_train = np.loadtxt(data_path+str(i)+'dice_tr.txt', dtype=float)
    dice_train = np.around(dice_train, decimals=4)
    x = np.squeeze(x, axis=1)
    prediction = prediction[:,:,:,:,1]
    flair = x[:,:,:,:,1]
    flair_train = (flair - flair.mean(axis=0)) / flair.std(axis=0)
    x_train = np.stack((prediction, flair_train), axis=4)
    ######################### LOAD validation DATA ######################
    with open(data_path+str(i)+'x_val.pkl', 'rb') as f:
        x = pickle.load(f)
    with open(data_path+str(i)+'prediction_val.pkl', 'rb') as f:
        prediction = pickle.load(f)
    dice_val = np.loadtxt(data_path+str(i)+'dice_val.txt', dtype=float)
    x = np.squeeze(x, axis=1)
    prediction = prediction[:,:,:,:,1]
    flair = x[:,:,:,:,1]
    flair_val = (flair - flair.mean(axis=0)) / flair.std(axis=0)
    x_val = np.stack((prediction, flair_val), axis=4)
    dice_val = np.around(dice_val, decimals=4)
    ######################### COMPILE THE MODEL ######################
    model = net.create_cnn(input_shape=input_shape)
    model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae'])
    print('Initiating training, input image-prediction pair...')

    train = [x_train, dice_train]
    val = [x_val, dice_val]
    ######################### SET UP DATA GENERATOR ######################
    genTra = gener.DataGenerator(list(range(len(x_train))), train, batch_size=2, shuffle=True)
    genVal = gener.DataGenerator(list(range(len(x_val))), val, batch_size=2, shuffle=True)
    
    checkpoint_IP = [CSVLogger('training_hist/'+str(i)+'training_IP.csv', append=True, separator=','),
                  ModelCheckpoint('models/'+str(i)+'best_model_IP.hdf5', monitor='val_loss', verbose=1, save_best_only= True, mode='min')]
    ######################### TRAIN THE MODEL ######################
    model.fit_generator(generator = genTra, epochs=100,steps_per_epoch=34, validation_data=genVal, validation_steps=8,callbacks=checkpoint_IP,use_multiprocessing=True)
    

