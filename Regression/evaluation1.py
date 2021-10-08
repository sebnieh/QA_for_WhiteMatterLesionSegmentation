'''
Evaluation script for regression models trained with uncertainty-prediction input and image-prediction  input.
'''
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import utils as ut
import numpy as np
import random
import pickle
import os 
import tensorflow as tf
import scipy.stats

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

data_path = '/opt/Lesion_Segmentation/reg_net/'

################### Evaluate models trained with uncertainty-prediction input and image-prediction input ###################

for i in np.arange(1,6):
    print('FOLD:', i)
    ######################### LOAD THE DATA ######################
    with open(data_path+str(i)+'x_test.pkl', 'rb') as f:
        x_t = pickle.load(f)
    with open(data_path+str(i)+'entropy_unc_test.pkl', 'rb') as f:
        entropy_map_t = pickle.load(f)
    with open(data_path+str(i)+'prediction_test.pkl', 'rb') as f:
        prediction_t = pickle.load(f)
    dice_test = np.loadtxt(data_path+str(i)+'dice_test.txt', dtype=float)
    x_t = np.squeeze(x_t, axis=1)
    prediction_t = prediction_t[:,:,:,:,1]
    entropy_map_t = entropy_map_t[:,:,:,:,1]
    flair_t = x_t[:,:,:,:,1]
    entropy_map_t = ut.rescale(entropy_map_t, max=1)
    flair_n = (flair_t - flair_t.mean(axis=0)) / flair_t.std(axis=0)
    dice_test = np.around(dice_test, decimals=4)
 
    x_test = np.stack((prediction_t, flair_n), axis=4)
    ####################### EVALUATION IMAGE-PREDICTION PAIR #######################
    model = load_model('models/'+str(i)+'best_model_IP.hdf5')
    predictions = model.predict(x_test)
    predictions = predictions.flatten()
    mae = mean_absolute_error(dice_test, predictions)
    print('MAE:', mae)
    corr = scipy.stats.pearsonr(dice_test, predictions)
    print('corr:', corr)
    rmse = np.sqrt(np.square(np.subtract(dice_test, predictions)).mean())
    print('RMSE:', rmse)
    np.savetxt('res/'+str(i)+'corr_IP.txt', np.array([corr]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'mae_IP.xt', np.array([mae]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'rmse_IP.xt', np.array([rmse]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'predictions_IP.txt', predictions, fmt='%1.9f')
    
    ####################### EVALUATION UNCERTAINTY-PREDICTION PAIR #######################

    x_test = np.stack((prediction_t, entropy_map_t), axis=4)
    
    model = load_model('models/'+str(i)+'best_model_PE.hdf5')
    predictions = model.predict(x_test)
    predictions = predictions.flatten()
    mae = mean_absolute_error(dice_test, predictions)
    corr = scipy.stats.pearsonr(dice_test, predictions)
    rmse = np.sqrt(np.square(np.subtract(dice_test, predictions)).mean())
    np.savetxt('res/'+str(i)+'corr_PE.txt', np.array([corr]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'mae_PE.txt', np.array([mae]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'rmse_PE.txt', np.array([rmse]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'predictions_PE.txt', predictions, fmt='%1.9f')
    