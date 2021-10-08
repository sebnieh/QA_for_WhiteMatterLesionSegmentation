"""
Functions used to preprocess the image data

"""


from os import walk
import matplotlib.pyplot as plt 
import nibabel as nib
import os 
import numpy as np
from skimage.transform import resize, downscale_local_mean, rescale
from skimage.restoration import denoise_nl_means, estimate_sigma
from random import randint
from scipy import ndimage as nd
from skimage import transform
import scipy.linalg as la
import scipy
import pandas as pd
from scipy import ndimage
from skimage import filters

##################### Data Loading #####################

def load_file(path_array):
        while True:
            try:
                volume = get_random_sample(path_array, batch_size=1)
                break
            except:
                continue
        return volume.transpose([1,2,3,4,0])
    
    
def get_data(path_list):
    array_temp = []
    for j in range(len(path_list)):
        img_temp = np.array(nib.load(path_list[j]).dataobj)
        array_temp.append(img_temp)
    return np.array(array_temp)

def get_random_sample(path_array, batch_size = None):
    '''
    Input:
        path_array:
            Two dimensional array,
            where the first dimension is the type
            of data and the second dimension the number of volumes

    Output:
        data_3D:
            Five dimensional array: First dimension, Volume type; Second dimension, Number of volumes;
            Three-Five, Volume shape
    '''
    indexes =  np.random.randint(0, len(path_array[0]), batch_size)
    data_3D = []
    for i in indexes:
        data_3D.append(get_data([path_array[0][i],path_array[1][i], path_array[2][i]]))
    return np.array(data_3D)

##################### Data Preprocessing #####################

def rescale(data, max=255):
    '''
    rescale array to [0 max]
    '''
    
    data = data-np.min(np.ravel(data))
    data = max* (data/np.max(np.ravel(data)))
    return data

def norm_label(y):
    '''
    Normalise the segmentation mask
    '''
    label = np.copy(y)
    for i in range(len(np.unique(label))):
        label[label == np.unique(label)[i]] = i
    return label

def prepare_y(y):
    '''
    Categorise segmentation GT map into 2 classes
    '''
    y = norm_label(y)
    if output_classes == 2:
        y[y>0] = 1
    return to_categorical(y, output_classes)


def thr_cuboid(vol, ker_size=5):
    '''
    threshold volume and get guboid
    '''

    # thresholding 
    val = filters.threshold_multiotsu(vol, 2)
    
    mask = np.zeros(vol.shape)
    mask[vol>val] = 1
    
    # mask processing 
    kernel=np.ones((ker_size, ker_size, ker_size), np.uint8)
    mask = ndimage.morphology.binary_closing(mask, structure=kernel)

    return mask

def get_cuboid(array, mask):
    '''
    Extracts from one volume the smallest cuboid containing all voxels that equal 1 in the mask
    :param array:   takes multidimensional array
    :param mask:    mask of 0s and 1s
    :return volumes:  cropped multidimensional array
    :return indices: indices of vol and lbl corresponding to cuboid
    '''
    new_array = []
    [xx, yy, zz] = np.nonzero(mask)
    [mx, Mx] = [min(xx), max(xx)]
    [my, My] = [min(yy), max(yy)]
    [mz, Mz] = [min(zz), max(zz)]
    indices = [mx, Mx, my, My, mz, Mz]

    for i in array:
        array_i = i[mx:Mx+1, my:My+1, mz:Mz+1]
        new_array.append(array_i)
    new_array = np.array(new_array)
    return new_array, indices
