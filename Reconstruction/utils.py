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
    # Inverse rescale using min and max value of original distribution 
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

    c_in  = 0.5 * np.array(data[1,:,:,:,0].shape)
    c_out = 0.5 * np.array(data[1,:,:,:,0].shape)
    offset = c_in - np.dot(invTransform, c_out)

    transformed = []
    
    for i in data[:,:,:,:,0]:
        vol = scipy.ndimage.interpolation.affine_transform(i, invTransform, offset=offset, output_shape=i.shape)
        transformed.append(vol)
    transformed = np.array(transformed)
    
    return transformed, invTransform


################## Noise ##################


def gen_patternNoise(data, max_m=10, max_s=7):
    '''
    Generates parameters to calculate space and frequency noise 
    NB: all generated parameters introduce some level of noise!

    Input:
    data  : input MRI volume
    max_m : maximum value of mean in a Gaussian distribution of noise to be added
    max_s : maximum value of std in a Gaussian distribution of noise to be added

    Output:
    sNoise_par, fNoise_par: arrays with parameters to calculate space and frequency noise

    '''
    
    # noise_param = np.zeros((11),dtype=int)
    fNoise_par  = np.zeros((7),dtype=int)
    sNoise_par  = np.zeros((4),dtype=int)
    nx,ny,nz    = np.shape(data)
    
    # gaussian noise
    sNoise_par[0]   = np.round(np.random.uniform(0,max_m,[1]))      
    sNoise_par[1]   = np.round(np.random.uniform(0,max_s,[1])) 
    # zipper
    sNoise_par[2]   = np.round(np.random.uniform(-nx+4,nx-4, [1]))
    sNoise_par[3]   = np.round(np.random.uniform(0,3, [1]))
    
    # gibbs ringing
    fNoise_par[0:2] = nx//12 + np.round(np.random.uniform(0,(nx//3),[2]))      
    # k-spike
    fNoise_par[2:4] = [nx//5, ny//5] + np.round(np.random.uniform(-(nx//16),(nx//16),[2]))  
    fNoise_par[4]   = np.round(np.random.uniform(2,20,[1]))
    # ghosting
    fNoise_par[5]   = np.round(np.random.uniform(1, 35, [1]))  
    fNoise_par[6]   = np.round(np.random.uniform(2,4, [1]))
    
    return sNoise_par, fNoise_par 


def add_freqNoise(data, fNoise_par, noiseFlags = [True, True, True]):
    '''
    Adds frequency-domain noise to data. 
    
    1. gibbs ringing    --> low-pass filter         --> (gx,gy)     
    2. k-spike          --> spike in k-space        --> (skx,sky,sk)  
    3. ghosting         --> undersampling in k-sp.  --> (c,k)       
    
    Input:
    data       : input MRI volume
    fNoise_par : array with parameters to calculate space and frequency noise
    noiseFlags : flags to determine if given noise is to be added [gibbs, spike, ghost]

    Output:
    noisy_data : volume with added noise

    '''

    gx,gy,skx,sky,sk,c,k = fNoise_par
    nx,ny,nz = np.shape(data)
    noisy_data = np.zeros([nx,ny,nz])
    
    c = np.random.choice([c, -c])
    
    # needs to do 2d analysis because 3d crashes!
    for z in np.arange(0,nz):
        
        f = np.fft.fft2(data[:,:,z],axes=(0,1))
        #print(str(z) + ' ' + str(np.shape(f)))
        f_shift = np.copy(f)
        f_shift = np.fft.fftshift(f_shift)
    
        msk = np.zeros([nx,ny])
        
        # 1. gibbs
        if noiseFlags[0]: 
            msk[gx:-gx,gy:-gy] = 1
        else:
            msk[:,:] = 1
        
        # 2. spike
        # sky>0: top-left --> bottom-right; sky<0: bottom-right --> top-left 
        if noiseFlags[1]: 
            msk[nx//2-skx-1:nx//2-skx+1,ny//2+sky-1:ny//2+sky+1] = sk
        
        # 3. ghosting
        if noiseFlags[2]:
            msk2 = np.abs(1-c/100)*np.ones([nx,ny])
            if c>0:
                msk2[::k,:] = 1
            else:
                msk2[:,::k] = 1
        else:
            msk2 = np.ones([nx,ny])

                
        msk = np.multiply(msk,msk2)
        f_shift = np.multiply(f_shift,msk)
        
        img_shift = np.fft.ifftshift(f_shift)    
        img_shift = np.fft.ifft2(f_shift,axes=(0,1))
        
        noisy_data[:,:,z] = np.abs(img_shift)
    
    noisy_data = np.uint16(ut.rescale(noisy_data))
    
    return noisy_data


def add_spaceNoise(data, sNoise_par, noiseFlags = [True, True]):
    '''
    Adds space-domain noise to data. 

    1. random noise     --> gaussian distribution   --> (m,s)
    2. zipper           --> noise line in space     --> (zk, w)     

    Input:
    data       : input MRI volume
    sNoise_par : array with parameters to calculate space and frequency noise
    noiseFlags : flags to determine if given noise is to be added [gaussian, zipper]

    Output:
    noisy_data : volume with added noise 

    '''
    
    m, s, zk, w = sNoise_par
    nx,ny,nz = np.shape(data[1,:,:,:])    
    ll = max(np.ravel(data[1,:,:,:]))*0.75
    hl = max(np.ravel(data[1,:,:,:]))*0.95
    

    noise      = np.random.normal(m, s, size=[nx,ny,nz])
    noisy_data = []
    for i in data[:,:,:,:]:
        vol = i+noise
        noisy_data.append(vol)

    return np.array(noisy_data)


def add_gaus_noise(data, lvl=0.8, noise_m=10, noise_s=8):
    '''
    Add space- and frequency-domain noise to data.
    Which particular type of noise to add is specified through boolean flags.

    Input:
    data    : input data (3D volumes)
    lvl     : probability that noise is added (which and to what extent is dealt with internally)
    noise_m : maximum value of mean for a gaussian distribution of noise N(mean,std)
    noise_s : maximum value of std for a gaussian distribution of noise N(mean,std)
    
    Ouput:
    Noisy data

    '''    
    
    sNoise_par, fNoise_par = gen_patternNoise(data[1,:,:,:], max_m=noise_m, max_s=noise_s)
    
    # boolean flags determining which particular type of noise to add
    tot_noiseFlags = np.random.choice([True, False], 5, p=[lvl, 1-lvl])

    noisy_data = add_spaceNoise(rescale(data, max=255), sNoise_par, noiseFlags = tot_noiseFlags[3::])

    return rescale(noisy_data, max=1)