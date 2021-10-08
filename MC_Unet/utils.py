"""
Script file with functions used to load and preprocess the data.
"""

import numpy as np
import nibabel as nib
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import filters
import pandas
import csv
from skimage import filters
import time

##################### Data Loading #####################

def get_data(path_list):
    array_temp = []
    for j in range(len(path_list)):
        img_temp = np.array(nib.load(path_list[j]).dataobj)
        img_temp = np.squeeze(img_temp)
        array_temp.append(img_temp)
    return np.array(array_temp)


def get_data_path_list(dataset_dir):
    t1, flair = [], []
    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = subdir + '/' + file # change to '\\' if using Windows
            if file_path.endswith("FLAIR.nii.gz"): 
                flair.append(file_path) 
            elif file_path.endswith("T1.nii.gz"): 
                t1.append(file_path) 
    return np.array([np.array(t1), np.array(flair)])

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


def get_wmls_path_list(dataset_dir):
    image_path_array1, image_path_array2, mask_path_array = [], [], []
    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = subdir + '/' + file # change to '\\' if using Windows
            if file_path.endswith("wmh.nii.gz"):
                mask_path_array.append(file_path)
            elif file_path.endswith("FLAIR.nii.gz"): # change to '\\' if using Windows
                image_path_array1.append(file_path)
            elif file_path.endswith("T1.nii.gz"): # change to '\\' if using Windows
                image_path_array2.append(file_path)   
    return np.array([np.array(image_path_array1),np.array(image_path_array2),np.array(mask_path_array)])

##################### Data Preprocessing #####################

def get_cuboid_DR(flair,t1,label, mask):
    '''
    Extracts from one volume the smallest cuboid containing all voxels that equal 1 in the mask
    :param array:   takes multidimensional array
    :param mask:    mask of 0s and 1s
    :return volumes:  cropped multidimensional array
    :return indices: indices of vol and lbl corresponding to cuboid
    '''
    [xx, yy, zz] = np.nonzero(mask)
    [mx, Mx] = [min(xx), max(xx)]
    [my, My] = [min(yy), max(yy)]
    [mz, Mz] = [min(zz), max(zz)]
    indices = [mx, Mx, my, My, mz, Mz]
    flair = flair[mx:Mx+1, my:My+1, mz:Mz+1]
    t1 = t1[mx:Mx+1, my:My+1, mz:Mz+1]
    label = label[mx:Mx+1, my:My+1, mz:Mz+1]
    
    return flair,t1, label, indices

def thr_cuboid(vol, ker_size=5):
    '''
    threshold volume and get cuboid
    '''
    #vol = vol[:,:,:,0]

    # thresholding 
    val = filters.threshold_otsu(vol, 2)

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
        
        

def get_cuboid_X(flair, t1, mask):
    '''
    Extracts from one volume the smallest cuboid containing all voxels that equal 1 in the mask
    :param array:   takes multidimensional array
    :param mask:    mask of 0s and 1s
    :return volumes:  cropped multidimensional array
    :return indices: indices of vol and lbl corresponding to cuboid
    '''
    [xx, yy, zz] = np.nonzero(mask)
    [mx, Mx] = [min(xx), max(xx)]
    [my, My] = [min(yy), max(yy)]
    [mz, Mz] = [min(zz), max(zz)]
    indices = [mx, Mx, my, My, mz, Mz]
    flair = flair[mx:Mx+1, my:My+1, mz:Mz+1]
    t1 = t1[mx:Mx+1, my:My+1, mz:Mz+1]
    
    return flair, t1, indices



##################### Data normalisation and rescaling ####################

def normalize_image(original_image):
    '''
    :param image_data: Four dimensional image array (image type, image number, x, y)
    :return: Normalized four dimensional image array (image type, image number, x, y)
    '''
    image_data = (original_image - np.min(original_image)) / np.max(original_image)
    return image_data

def rescale(data, max=255):
    '''
    rescale array to [0 max]
    '''
    
    data = data-np.min(np.ravel(data))
    data = max* (data/np.max(np.ravel(data)))
    return data