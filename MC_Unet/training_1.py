"""
Training script for folds 1-3.
"""
import MCD_unet
import generator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from loss_functions import dice_loss, dice_score, dice_coef
from tensorflow.keras.optimizers import Adam
import os
from sklearn.model_selection import KFold
import utils as ut
import pandas as pd
import numpy as np


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple


input_shape = (128, 128, 128, 2)
nr_of_classes = 2
learning_rate = 0.001
num_folds = 5
batch_size = 1
EPOCHS = 300
steps_per_e = 10
validation_steps = 1

files = '/home/elena/mcd'
aug_files = '/home/elena/mcd'
dataset_dir = '/opt/Lesion_Segmentation/data/wmls/'


loader = generator.data_loader(files, "utils", x_dim = input_shape[0], y_dim = input_shape[1], z_dim = input_shape[2], b=10, aug_dir = aug_files, dataset_dir = dataset_dir, aug_file = "main_augmentation", output_classes = nr_of_classes)
print("Initialized loader...")


path_array = ut.get_wmls_path_list(dataset_dir) # read in path arrays
print(path_array.shape)

# 3-fold Cross Validation model evaluation
fold_no = 1
for i in np.arange(1,3):
    train_ids = np.loadtxt('/home/elena/mcd/ids/'+str(i)+'_train_ids.txt', dtype=int)
    val_ids = np.loadtxt('/home/elena/mcd/ids/'+str(i)+'_val_ids.txt', dtype=int)

    print('\nFold ', fold_no)

    train_cv = path_array[:,train_ids]
    valid_cv = path_array[:,val_ids]
    x,y = loader.get_valset(valid_cv)
    print("Loaded validation data...", x.shape, y.shape)



    model = MCD_unet.MCD_UNet(input_shape = input_shape, outputChannel = nr_of_classes)
    model.compile(loss = dice_loss, optimizer=Adam(lr = learning_rate),  metrics = [dice_score])

    print("Compiled model...")

    file = str(i) + "_fold_best_model.hdf5"
    filepath = str(i) + "_fold_model_{epoch:03d}.hdf5"    
    checkpoint = [CSVLogger('training_1.csv', append=True, separator=','),
                  ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only= True, mode='min')]

    model.fit_generator(generator = loader.generate_volume(train_cv), validation_data = (x,y),
                    steps_per_epoch = steps_per_e, epochs = EPOCHS, callbacks = checkpoint)
    
    
    # Increase fold number
    fold_no = fold_no + 1

