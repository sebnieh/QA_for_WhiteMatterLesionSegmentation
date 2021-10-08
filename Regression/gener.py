'''
Data generator.
Also handels preprocessing and augmentation.

For reference see:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
'''

import numpy as np
import tensorflow.keras as tfk
import nibabel as nib
from skimage.transform import rescale, resize, downscale_local_mean
import utils as ut
import pickle
import random

class DataGenerator(tfk.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        
        self.on_epoch_end()
        

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        

        if self.shuffle == True:
            random.sample(self.list_IDs, len(self.list_IDs))
            
            
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index): # FUNCTION TO PICK UNIQUE INDEXES PER 1 EPOCH
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        

        list_IDs_temp   = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp, self.labels)

        return X, y


    def __data_generation(self, ids, labels):
        '''
        Generates preprocessed and augmented data containing batch_size samples ;  X : (n_samples, *dim, n_channels)
        
        Parameters
        ----------
        list_IDs_temp   : list of indices
        labels_temp     : list of lists: labels_temp[0..Nch-1] = filenames of data; labels_temp[Nch] = filenames of labels;

        Returns
        -------
        X               : data in 5D format   (self.batch_size, self.dim[0], self.dim[1], self.dim[2], Nch)
        y               : labels in 4D format (self.batch_size, self.dim[0], self.dim[1], self.dim[2], Ncl)

        '''

        x_array, y_array = [], []
        
        for i in range(len(ids)):
            
            sample = labels[0][ids[i]]
            
            sample = sample.transpose([3,0,1,2])
            
            y = labels[1][ids[i]]
            x, affMAt = ut.affine_tr(sample, 0.7, max_rot=10) # PERFORM AFFINE TRANSFORMATIONS
            x = x.transpose([1,2,3,0])
            
            x_array.append(x)
            y_array.append(y)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
       
        return x_array, y_array
