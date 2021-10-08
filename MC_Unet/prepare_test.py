"""
Test data preparation: pickles the prediction, ground truth segmentation, FLAIR and uncertainty maps, saves all of the dice results per fold.
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
dataset_dir = '/opt/Lesion_Segmentation/data/wmls/elena_thesis'


loader = generator2.data_loader(files, "utils", x_dim = input_shape[0], y_dim = input_shape[1], z_dim = input_shape[2], b=10, aug_dir = aug_files, dataset_dir = dataset_dir, aug_file = "main_augmentation", output_classes = nr_of_classes)
print("Initialized loader...")


path_array = ut.get_wmls_path_list(dataset_dir) # read in path arrays
print(path_array.shape)


# 5-fold TEST DATA PREPARATION OF PREDICTION MAPS, FLAIR MAPS AND UNCERTAINTY MAPS AND DICE VALUES
fold_no = 1
for i in np.arange(1,6):
    
    test_ids = np.loadtxt('/home/elena/mcd/ids/'+str(i)+'_test_ids.txt', dtype=int)

    print('\nFold ', i)

    test_cv = path_array[:,test_ids]
    
    x_test, y_test = loader.get_data(test_cv)
    print("Loaded test data...", x_test.shape, y_test.shape)

    model = load_model('/home/elena/mcd/models/'+str(i)+'_fold_best_model.hdf5', custom_objects={'dice_loss': dice_loss, 'dice_score':dice_score, 'InstanceNormalization':InstanceNormalization})

    print("Loaded model...")
    
    prediction_test, entropy_unc_test, dice_values_test = [], [], []
    print("Prepare validation data")
    
    for k in range(x_test.shape[0]):
        pred, unc_map, dice = ut.entropy_calc(x_test[k], y_test[k], model, 1, sample_times=20)
        prediction_test.append(pred)
        entropy_unc_test.append(unc_map)
        dice_values_test.append(dice)
    
    prediction_test = np.array(prediction_test)
    entropy_unc_test = np.array(entropy_unc_test)
    dice_values_test = np.array(dice_values_test)
        
    with open("/opt/"+str(i)+'x_test.pkl','wb') as f:
        pickle.dump(x_test, f)
    with open("/opt/"+str(i)+'prediction_test.pkl','wb') as f:
        pickle.dump(prediction_test, f)
    with open("/opt/"+str(i)+'entropy_unc_test.pkl','wb') as f:
        pickle.dump(entropy_unc_test, f) 

    np.savetxt("/opt/"+str(i)+'dice_test.txt', dice_values_test, fmt='%1.9f')

    
    # Increase fold number
    fold_no = fold_no + 1

