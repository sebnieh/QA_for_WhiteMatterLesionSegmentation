"""
Data preparation: pickles the prediction, ground truth segmentation, FLAIR and uncertainty maps, saves all of the dice results per fold.
"""

import MCD_unet
import generator2
from loss_functions import dice_loss, dice_score, dice_coef
import os
from sklearn.model_selection import KFold
import utils as ut
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import load_model
from tfk_instance_norm import InstanceNormalization
import pickle 


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

# Parameters

nr_of_classes = 2
input_shape = (128, 128, 128, 2)
x_dim = input_shape[0] 
y_dim = input_shape[1]
z_dim = input_shape[2]

num_folds = 5

files = '/home/elena/mcd'
aug_files = '/home/elena/mcd'
dataset_dir = '/opt/Lesion_Segmentation/data/wmls/'


loader = generator2.data_loader(files, "utils", x_dim = input_shape[0], y_dim = input_shape[1], z_dim = input_shape[2], b=10, aug_dir = aug_files, dataset_dir = dataset_dir, aug_file = "main_augmentation", output_classes = nr_of_classes)
print("Initialized loader...")


path_array = ut.get_wmls_path_list(dataset_dir) # read in path arrays
print(path_array.shape)


# 5-fold DATA PREPARATION OF PREDICTION MAPS, FLAIR MAPS AND UNCERTAINTY MAPS AND DICE VALUES

fold_no = 1
for i in np.arange(1,6):
    
    train_ids = np.loadtxt('/home/elena/mcd/ids/'+str(i)+'_train_ids.txt', dtype=int)
    val_ids = np.loadtxt('/home/elena/mcd/ids/'+str(i)+'_val_ids.txt', dtype=int)

    print('\nFold ', i)

    train_cv = path_array[:,train_ids]
    valid_cv = path_array[:,val_ids]
    
    x_tr, y_tr = loader.get_data(train_cv)
    print("Loaded training data...", x_tr.shape, y_tr.shape)
    x_v, y_v = loader.get_data(valid_cv)
    print("Loaded validation data...", x_v.shape, y_v.shape)
    
    model = load_model('/home/elena/mcd/models/'+str(i)+'_fold_best_model.hdf5', custom_objects={'dice_loss': dice_loss, 'dice_score':dice_score, 'InstanceNormalization':InstanceNormalization})

    print("Loaded model...")
    
    prediction_val, entropy_unc_val, dice_values_val = [], [], []
    print("Prepare validation data")
    
    for k in range(x_v.shape[0]):
        pred, unc_map, dice = ut.entropy_calc(x_v[k], y_v[k], model, 1, sample_times=20)
        prediction_val.append(pred)
        entropy_unc_val.append(unc_map)
        dice_values_val.append(dice)
    
    prediction_val = np.array(prediction_val)
    entropy_unc_val = np.array(entropy_unc_val)
    dice_values_val = np.array(dice_values_val)
    
    print(dice_values_val)
    
    with open("/opt/"+str(i)+'x_val.pkl','wb') as f:
        pickle.dump(x_v, f)
    with open("/opt/"+str(i)+'prediction_val.pkl','wb') as f:
        pickle.dump(prediction_val, f)
    with open("/opt/"+str(i)+'entropy_unc_val.pkl','wb') as f:
        pickle.dump(entropy_unc_val, f) 

    np.savetxt("/opt/"+str(i)+'dice_val.txt', dice_values_val, fmt='%1.9f')

    prediction_tr, entropy_unc_tr, dice_values_tr = [], [], []
    
    print("Prepare training data")
    
    for k in range(x_tr.shape[0]):
        pred, unc_map, dice = ut.entropy_calc(x_tr[k], y_tr[k], model, 1, sample_times=20)
        prediction_tr.append(pred)
        entropy_unc_tr.append(unc_map)
        dice_values_tr.append(dice)
    
    prediction_tr = np.array(prediction_tr)
    entropy_unc_tr = np.array(entropy_unc_tr)
    dice_values_tr = np.array(dice_values_tr)
    
    
    with open("/opt/"+str(i)+'x_train.pkl','wb') as f:
        pickle.dump(x_tr, f)
    with open("/opt/"+str(i)+'prediction_tr.pkl','wb') as f:
        pickle.dump(prediction_tr, f)
    with open("/opt/"+str(i)+'entropy_unc_tr.pkl','wb') as f:
        pickle.dump(entropy_unc_tr, f) 
    np.savetxt("/opt/"+str(i)+'dice_tr.txt', dice_values_tr, fmt='%1.9f')

    # Increase fold number
    fold_no = fold_no + 1

