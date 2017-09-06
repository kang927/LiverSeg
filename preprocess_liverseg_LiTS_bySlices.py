# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:13:03 2017

@author: kang927

script to train sagittal slices of the CT volumes


"""

# preprocess the slices 

import numpy as np
import os

# for visual displlay of vollume
from matplotlib import pyplot as plt
from IPython import display

from scipy.misc import imresize,imsave

import nibabel as nib # for reading nifTi data file 



# define const variable
IMG_DTYPE = np.float32
SEG_DTYPE = np.int16
HU_min = -100 # air = -1000 HU, water = 0 HU, fat is -120 to -90
HU_max = 400 # practical speaking soft-tissue contrast 100 to 300
IMAGE_SIZE = 256
# pad 20% around the image, so when perform data augmentation will be ok
PAD_SIZE = 48

#%%
def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage: 
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    plt.show()


def find_liver_slice(datavol,segvol):
    """
    for training purpose we will use only train on slices that have the liver
    segmentation 
    
    """
    ind_liver = (np.sum(segvol,axis=(0,1)) != 0)
    datavol = datavol[:,:,ind_liver]
    segvol = segvol[:,:,ind_liver]
    return datavol,segvol



def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)



def process_img_slice(img_slc):
    img_slc   = img_slc.astype(IMG_DTYPE)
    img_slc[img_slc>1000] = 0 # remove artifact type of HU, not significant for soft-tissue interpretation
    img_slc   = np.clip(img_slc, HU_min, HU_max) # HU units
    img_slc   = imresize(img_slc, (IMAGE_SIZE,IMAGE_SIZE),interp="nearest")
    #img_slc   = np.pad(img_slc,((PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE)),mode='constant')
    
    return img_slc


def preprocess_data_vol(datavol,segvol):
    """ process the whole image volume rather than slice by slice"""
    
    # first we will only use those slices with liver volume
    datavol,segvol = find_liver_slice(datavol,segvol)
    Nz = datavol.shape[2]
    datavol2 = np.zeros((IMAGE_SIZE,IMAGE_SIZE,Nz))
    segvol2 = np.zeros((IMAGE_SIZE,IMAGE_SIZE,Nz))
    for s in range(Nz):
       datavol2[:,:,s] = process_img_slice(datavol[:,:,s])
       segvol2[:,:,s] = imresize(segvol[:,:,s],(IMAGE_SIZE,IMAGE_SIZE),interp='nearest',mode='F') # when using resize
       # need to use floating pt mode otherwise will map 1 to 255
    # end slice by slice processing
    datavol2 = datavol2/1000 # HU is basically 10000x mu - mu_water/(mu_water-mu_air)
    # no need for augmentation at this point
    #datavol2 = np.pad(datavol2,((PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
    #segvol2 = np.pad(segvol2,((PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
    
    return datavol2, segvol2



def check_binaryImage(seg_vol):
    """ simple logical operation to make sure the segmentation map is (0,1)"""
    nvol = np.sum( np.logical_and(seg_vol.flatten()!=0,seg_vol.flatten()!=1) )
    print("number of elements that are not 0 or 1 are:" + str(nvol) +"\n")
    return nvol

# helper function the full path of a list of files in a directory
def listdir_fullpath(dn):
    return [os.path.join(dn,fn) for fn in os.listdir(dn)]
            
def orient_dicom(dataobj):
    """ 
    based on the affine transform of the header convert the 
    array to appropriate DICOM orientation
    Argument:
        nifti image data object from nibabel
    
    Output:
        3D numpy array of DICOM orientation
    """
    M = dataobj.affine[:3,:3]
    x_flag = M[0,0]
    y_flag = M[1,1]
    z_flag = M[2,2]
    # DICOM use (column,row) rather than the conventional matrix orientation
    tmp = np.transpose(dataobj.get_data(),(1,0,2))
    if x_flag > 0:
        tmp = np.fliplr(tmp)
    
    if y_flag > 0:
        tmp = np.flipud(tmp)
    
    if z_flag > 0:
        tmp = tmp[:,:,::-1] # reverse its dimension

    return tmp

    
def save_img_slice(datavol,file_prefix):
    nslice = datavol.shape[2]
    for s in range(nslice):
        fn = file_prefix + '_slice_' + str(s) + '.npy'
        np.save( fn,datavol[:,:,s])
#%%
preprocess_data = True


if preprocess_data:
    dataDir='D:/liverseg_training/training_data_liverseg/'
    # data is too big to fit in memory, need to write it out to directoy
    train_img_dir = 'D:/liverseg_training/training_slice_sagittal/images/'
    train_seg_dir = 'D:/liverseg_training/training_slice_sagittal/segmentations/'
    Nvol=131
    for s in range(0,Nvol):
        ii = s
        datafn = dataDir + 'volume-'+ str(ii) + '.nii'
        segfn = dataDir + 'segmentation-'+ str(ii) + '.nii'
        img = nib.load(datafn)     
        seg = nib.load(segfn)
        # make the data has the correct radiological view order
        img_data = orient_dicom(img)
        seg_data = orient_dicom(seg)
        seg_data[seg_data!=0]=1 # set all non-zero label to 1
        
        # redorder dimension so the data is sagittal (z,y,x=slice)
        img_data = np.transpose(img_data,[2,0,1])
        seg_data = np.transpose(seg_data,[2,0,1]) 
          
        # permute the data volume so it is stacked at sagittal slices
        img_data,seg_data = preprocess_data_vol(img_data,seg_data)
                
        # package the data in the order for keras (N_sample x height x width x N_channels)
        # this is only for preprocessing data
        nx=img_data.shape[0]
        ny=img_data.shape[1]
        nz=img_data.shape[2] 
        # check whether segmentation file is only (0,1)
        check_binaryImage(seg_data)        
        # save per slice so can train with batch of data        
        img_prefix=train_img_dir + 'image' + str(ii)
        seg_prefix=train_seg_dir + 'seg' + str(ii)
        save_img_slice(img_data,img_prefix)
        save_img_slice(seg_data,seg_prefix)
# end preprocess


