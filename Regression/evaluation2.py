'''
Evaluation script for regression models trained with error-prediction input
'''
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import utils_reg as ut
import numpy as np
import random
import pickle
import os 
import tensorflow as tf
import scipy.stats

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

data_path = '/opt/Lesion_Segmentation/reg_net/'

################### Evaluate models trained with error-prediction input ###################

for i in np.arange(1,6):
    print('FOLD:', i)
    print("Loading the data...")
    ######################### LOAD THE DATA ######################

    with open(data_path+str(i)+'em.pkl', 'rb') as f:
        x_test = pickle.load(f)
    dice_test = np.loadtxt(data_path+str(i)+'dice_test.txt', dtype=float)
    dice_test = np.around(dice_test, decimals=4)
    x_test = ut.rescale(x_test, max=1)
    x_test = np.squeeze(x_test)
    with open(data_path+str(i)+'prediction_test.pkl', 'rb') as f:
        prediction = pickle.load(f)

    prediction = prediction[:,:,:,:,1]
    x_test = np.stack((prediction, x_test), axis=4)

    print(x_test.shape)
    ####################### EVALUATION ERROR-PREDICTION PAIR #######################

    model = load_model('models/'+str(i)+'best_model_EM.hdf5')
    predictions = model.predict(x_test)
    predictions = predictions.flatten()
    mae = mean_absolute_error(dice_test, predictions)
    print('MAE:', mae)
    corr = scipy.stats.pearsonr(dice_test, predictions)
    print('corr:', corr)
    rmse = np.sqrt(np.square(np.subtract(dice_test, predictions)).mean())
    print('RMSE:', rmse)
    np.savetxt('res/'+str(i)+'corr_em.txt', np.array([corr]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'mae_em.txt', np.array([mae]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'rmse_em.txt', np.array([rmse]), fmt='%1.9f')
    np.savetxt('res/'+str(i)+'predictions_em.txt', predictions, fmt='%1.9f')
    
    
    
####################### ESTIMATE MEAN MAE AND CORR COEFFICIENTS #######################

final_res = []    
for i in np.arange(1,6):
    print('FOLD:', i)
    final_res.append(np.loadtxt('res/'+str(i)+'mae_em.txt', dtype=float))
    
final_res = np.array(final_res)
final_res = np.mean(final_res)
print('MEAN MAE:', final_res)


final_res = []
for i in np.arange(1,6):
    final_res.append(np.loadtxt('res/'+str(i)+'corr_em.txt', dtype=float))

final_res = np.array(final_res)
final_res = np.mean(final_res)
print('MEAN CORR:', final_res)


with open(str(1)+'pred.pkl','wb') as f:
    pickle.dump(predictions, f)
