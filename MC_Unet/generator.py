"""
This script allows to load and preprocess MRI data 
"""

import sys
import numpy as np
from tensorflow.keras.utils import to_categorical
from skimage.transform import rescale, resize
from keras.callbacks import Callback
import nibabel as nib


class data_loader:
    def __init__(self, utils_dir, utils_file, x_dim ,y_dim, z_dim, b, aug_dir, aug_file, dataset_dir, output_classes):
        self.utils = self.import_file(utils_dir, utils_file)
        self.augment = self.import_file(aug_dir, aug_file)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.b = b
        self.output_classes = output_classes

########## Helpful functions ##########

    def import_file(self, file_path, file_name):
        sys.path.insert(1, file_path)
        return __import__(file_name)

    def norm_label(self, y):
        label = np.copy(y)
        for i in range(len(np.unique(label))):
            label[label == np.unique(label)[i]] = i
        return label
    
    def prepare_y(self, y):
        y = self.norm_label(y)
        if self.output_classes == 2:
            y[y>0] = 1
        return to_categorical(y, self.output_classes)
    
    def load_file(self, path_array):
        while True:
            try:
                volume = self.utils.get_random_sample(path_array, batch_size=2)
                break
            except:
                continue
        return volume.transpose([1,2,3,4,0])

########## Data Generation ##########

    def get_valset(self, path_array): # Generate the validation set
        x_data, y_data = [], []
        for _ in range(path_array.shape[1]):
            volume = self.load_file(path_array)
            volume[2][volume[2]>0] = 1
            
            # cropping
            mask = self.utils.thr_cuboid(volume[0])
            volume, indx = self.utils.get_cuboid(volume, mask) 
            volume, affMAt = self.augment.augment(volume)
            # bi-cubic interpolation
            x = resize(volume[0],[self.x_dim, self.y_dim, self.z_dim,1],order=3,anti_aliasing=True)
            x_t1 = resize(volume[1],[self.x_dim, self.y_dim, self.z_dim,1],order=3,anti_aliasing=True)
            x = np.stack((x_t1,x), axis=3)
            x = np.squeeze(x, axis=4)   
            x = self.utils.rescale(x,max=255) # intensity normalisation
            
            y = resize(np.array(volume[2]),[self.x_dim, self.y_dim, self.z_dim,1],order=3, preserve_range=True)
            th = 0.3
            y[y>=th] = 1
            y[y<th] = 0
            y  = self.prepare_y(y)
            x_data.append(x)
            y_data.append(y)
        return np.array(x_data), np.array(y_data)
    


    def generate_volume(self, path_array): # generation of data dynamically
        while True:
            volume = self.load_file(path_array)
            volume[2][volume[2]>0] = 1
            # cropping
            
            mask = self.utils.thr_cuboid(volume[0])
            volume, indx = self.utils.get_cuboid(volume, mask)
            volume, affMAt = self.augment.augment(volume)
            # bi-cubic interpolation
            x = resize(volume[0], [self.x_dim, self.y_dim, self.z_dim, 1], order=3, anti_aliasing=True)
            x_t1 = resize(volume[1], [self.x_dim, self.y_dim, self.z_dim, 1], order=3, preserve_range=True)
            x = np.stack((x_t1,x), axis=3)
            x = np.squeeze(x, axis=4)
            x = self.utils.rescale(x,max=255) # intensity normalisation
            
            y = resize(np.array(volume[2]), [self.x_dim, self.y_dim, self.z_dim, 1], order=3, anti_aliasing=True)
            th = 0.3
            y[y>=th] = 1 # threschold the segmentation map after augmentation
            y[y<th] = 0
            y  = self.prepare_y(y)
            x = np.expand_dims(x, axis=0) # add batch dimension
            y = np.expand_dims(y, axis=0) # add batch dimension
            yield x, y




