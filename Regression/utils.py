"""
Script file with functions used to load and preprocess the data 
"""
import numpy as np
import os
from random import randint
import matplotlib.pyplot as plt
import os
import scipy
from scipy import ndimage as nd
from skimage import transform
import scipy.linalg as la
from skimage.transform import resize

def y_scaler(x, max_val=1):
    """Scale between 0 and 1
    x: data to be scaled,
    max_val: maximum value of dice, default=1."""
    min_val = np.nanmean(x, axis=0)
    x_std = (x - min_val) / (max_val - min_val)
    return x_std, min_val, max_val


def y_inv_scaler(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def rescale(data, max=255):
    '''
    rescale array to [0 max]
    '''
    
    data = data-np.min(np.ravel(data))
    data = max* (data/np.max(np.ravel(data)))
    return data



################ Affine transformations ##################

####### Rotation ########


def rot_x(a,R=None):
    
    B = np.array([ [1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)] ])
    
    if R is None:
        R = B
    else:
        R = np.dot(R,B)
    
    return R
    
def rot_y(a,R=None):
    B = np.array([ [np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)] ])
    
    if R is None:
        R = B
    else:
        R = np.dot(R,B)
    
    return R

def rot_z(a,R=None):
    B = np.array([ [np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1] ])
        
    if R is None:
        R = B
    else:
        R = np.dot(R,B)
    
    return R

def rotMat(ang,R=None):
    B = rot_x(ang[0])
    B = rot_y(ang[1],B)
    B = rot_z(ang[2],B)
        
    if R is None:
        R = B
    else:
        R = np.dot(R,B)
    
    return R

######## Scaling ########

def scale_mat(sc_arr,R=None):
    B = np.diag(sc_arr[0])
    if R is None:
        R = B
    else:
        R = np.dot(R,B)
    return R

######## Shearing ########

def shear_mat(sh_arr,R=None):
    i = np.array([1.,1.,1.])
    B = np.diag(i)
    B[0][1:3] = sh_arr[0][0:2]
    B[1][2]   = sh_arr[0][2]
    B[1][0]   = sh_arr[0][3]
    B[2][0:2] = sh_arr[0][0:2]
    
    if R is None:
        R = B
    else:
        R = np.dot(R,B)    
    
    return R

def comb_matr(mat_list):
    
    M = np.diag(np.array([1.,1.,1.]))
    for it in mat_list:
        M = np.dot(M, it)
    
    return M

###### Application of combined affine transformations ######

def affine_tr(data, lvl=0.7, max_rot=10, max_scal=0.2, max_shear=0.2):
    """
    Input:
    data      : input data (3D volumes)
    label     : labels relative to the MRI volume 
    lvl       : probability that noise is added (which and to what extent is dealt with internally)
    max_rot   : maximum amount of rotation (null=0)  
    max_scal  : maximum amount of scaling (null=0)
    max_shear : maximum amount of shearing (null=0)
    
    Output:
    aff_data  : transformed 3D volume
    aff_label : transformed labels
    invTransform: matrix used for transformation
    """
    
    
    tot_affFlags = np.random.choice([True, False], 3, p=[lvl, 1-lvl])
    
    angs = np.random.randint(-max_rot, max_rot, [1, 3])     
    angs = angs*np.pi/180.0                                 # angles in radiants
    sc_arr = np.ones([1, 3])  + max_scal*np.random.rand(1, 3) 
    sh_arr = np.zeros([1, 6]) + max_shear*np.random.rand(1, 6)    

    # matrices
    
    mat_list = []
    
    if tot_affFlags[0]: 
        mat_list.append(la.inv(rotMat(angs[0])))
        
    if tot_affFlags[1]: 
        mat_list.append(la.inv(scale_mat(sc_arr)))
        
    if tot_affFlags[2]:     
        mat_list.append(la.inv(shear_mat(sh_arr)))
    
    invTransform = comb_matr(mat_list)

    c_in  = 0.5 * np.array(data[0,:,:,:].shape)
    c_out = 0.5 * np.array(data[0,:,:,:].shape)
    offset = c_in - np.dot(invTransform, c_out)

    transformed = []
    
    for i in data[:]:
        vol = scipy.ndimage.interpolation.affine_transform(i, invTransform, offset=offset, output_shape=i.shape)
        transformed.append(vol)
    transformed = np.array(transformed)
    
    return transformed, invTransform


