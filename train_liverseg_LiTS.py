#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 11:36:21 2017

@author: kang927

script to setup training data and data augmentation generator 
for training of liver segmentation

data is from MICCAI 2017 LiTS challenge consist of 120 abdominal CT scans
for training

"""

import numpy as np
import os

# for visual displlay of vollume
from matplotlib import pyplot as plt
from IPython import display

#from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
#from unet import UNet, preprocess_input, dice_coef, dice_coef_loss
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize,imsave

import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)


import nibabel as nib # for reading nifTi data file 
from unet import UNet, dice_coef_loss, dice_coef


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
    datavol2 = np.pad(datavol2,((PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
    segvol2 = np.pad(segvol2,((PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
    
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
        
        
    
#%%
preprocess_data_train=False
preprocess_data_test=False
Ntrain = 100
Ntest = 30
# random split the data volumes to training and test/dev set
setInd = np.random.randint(0,130,130)

if preprocess_data_train:
    dataDir='D:/liverseg_training/training_data_liverseg/'
    # data is too big to fit in memory, need to write it out to directoy
    train_img_dir = 'C:/Users/kang927/Documents/deep_learning_liverseg/training/images/'
    train_seg_dir = 'C:/Users/kang927/Documents/deep_learning_liverseg/training/segmentations/'
    
    for s in range(90,131):
        ii = s
        datafn = dataDir + 'volume-'+ str(ii) + '.nii'
        segfn = dataDir + 'segmentation-'+ str(ii) + '.nii'
        img = nib.load(datafn)     
        seg = nib.load(segfn)
        # make the data has the correct radiological view order
        img_data = np.flipud( np.transpose(img.get_data(),(1,0,2)) )
        seg_data = np.flipud( np.transpose(seg.get_data(),(1,0,2)) )
        seg_data[seg_data!=0]=1 # set all non-zero label to 1
        
        # preprocess the data 
        img_data,seg_data = preprocess_data_vol(img_data,seg_data)
        
        # this is only for preprocessing data
        nx=img_data.shape[0]
        ny=img_data.shape[1]
        nz=img_data.shape[2]
        # package the data in the order for keras
        img_data = np.reshape(np.transpose(img_data,[2,0,1]),(nz,nx,ny,1))
        seg_data = np.reshape(np.transpose(seg_data,[2,0,1]),(nz,nx,ny,1)) 
        # check whether segmentation file is only (0,1)
        check_binaryImage(seg_data)
        imgfn=train_img_dir + 'image' + str(ii) + '.npy'
        segfn=train_seg_dir + 'seg' + str(ii) + '.npy'
        np.save(imgfn,img_data)
        np.save(segfn,seg_data)

# end preprocess


#%%
# since memory is limit, will need to load batch of data using generator
# batch_generator:
# infinite loop
# shuffle the list of files
# for each slice of the shuffled files where len(slice) == batch_size
#           open file and read a single array wit shape[0]== batch size --> yield data
#           have an edge case to handle the case where batch_size is not multiple of N_batches


def batch_generator(img_seg_generator,
                    seed,
                    data_list,
                    batch_size,
                    batch_vol=1):
    """ 
    generator that iterate through batch of volume data and generate
    augmented 2D images for training
    
    Argument:
        img_seg_generator: keras ImageDataGenerator object for images and segmentations 
        data_list: a list contain the directory of data volume in .npy format
        batch_vol: how many volume data to load at each iteration
        batch_size: number of images to generate per iteration
    """
    Nvol = len(data_list[0]) # number of data volumes
    ii=0
    
    while 1:
       # print("using data vol #" + str(ii) + "\n")
        # load img and segmentation        
        img = np.load(data_list[0][ii])
        seg = np.load(data_list[1][ii])
        # sainity check see if the img and segmentation result has the same dimension
        if not all( list(i == j for i,j in zip(img.shape,seg.shape)) ):
            raise ValueError('Dimension mismatch: image has dimension: ' + str(img.shape) + "\n" +
                             'segmentation has dimension: ' + str(seg.shape) + "\n")
        
      #  save_img_dn = 'C:/Users/kang927/Documents/deep_learning_liverseg/preview/images/'
      #  save_mask_dn= 'C:/Users/kang927/Documents/deep_learning_liverseg/preview/segmentations/'

        img_generator = img_seg_generator[0].flow(img,
                                    seed=seed,
                                    batch_size=batch_size
                                    )

        seg_generator = img_seg_generator[1].flow(seg,
                                    seed=seed,
                                    batch_size=batch_size                                    
                                    )
        ct = 0;
        for x_train,y_train in zip(img_generator,seg_generator):
            if ct >0:
                break
            ct+=1
        # end generate 1 iteration of the generator
        
        # advance the index to next volume 
        ii=ii+1
        if ii%Nvol ==0:
            #print("reseting ii to 0\n")
            ii=0 # if we loop through all data vol, repeat again
            
        yield x_train, y_train


#%%
train_img_dir = 'C:/Users/kang927/Documents/deep_learning_liverseg/training/images/'
train_seg_dir = 'C:/Users/kang927/Documents/deep_learning_liverseg/training/segmentations/'
    
list_img = listdir_fullpath(train_img_dir)
list_seg = listdir_fullpath(train_seg_dir)


# spliting into training and test data 
#randInd = np.random.randint(low=0,high=130)
test_end = 10
test_img = list_img[0:test_end]
test_seg = list_seg[0:test_end]
train_img = list_img[test_end:]
train_seg = list_seg[test_end:]
data_list = (train_img,train_seg)

#%%
fn = test_img[0]
x_test = np.load(fn)
fn = test_seg[0]
y_test = np.load(fn)

# obtaining the testing set 
n=len(test_img)
for ii in range(1,10):
    fn = test_img[ii]
    tmp = np.load(fn)
    x_test = np.concatenate( (x_test,tmp),axis=0)
    fn = test_seg[ii]
    tmp = np.load(fn)
    y_test = np.concatenate( (y_test,tmp),axis=0)

#%%

#%%
# setup the ImageDataGenerator
seed = 0
#data_gen_args = dict(
#        rotation_range=40,
#        width_shift_range=0.1,
#        height_shift_range=0.1,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        elastic_transform=True,
#        fill_mode='nearest',
#        data_format="channels_last")

data_gen_args=dict(data_format="channels_last")


image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
image_datagen.fit(x_test, augment=True, seed=seed)
mask_datagen.fit(y_test, augment=True, seed=seed)
data_gen = (image_datagen,mask_datagen)

# make the test generator
testimg_datagen=ImageDataGenerator(data_format="channels_last")
testseg_datagen=ImageDataGenerator(data_format="channels_last")
testimg_datagen.fit(x_test, augment=False,seed=seed)
testseg_datagen.fit(y_test, augment=False,seed=seed)
testimg_generator = testimg_datagen.flow(x_test,seed=seed)
testseg_generator = testseg_datagen.flow(y_test,seed=seed)
test_generator = zip(testimg_generator,testseg_generator)

batchgen = batch_generator(data_gen,seed, data_list,batch_size=4)


#%%
nx = x_test.shape[1]
ny = x_test.shape[2]
n_channels = x_test.shape[3]

model = UNet( input_shape=(nx,ny,n_channels))
model.compile(optimizer=Adam(lr=1e-3,decay=0.9), loss=dice_coef_loss, metrics=[dice_coef])


#%%
num_epochs=10
hist1 = model.fit_generator(batchgen,steps_per_epoch=4000,epochs=num_epochs,
                               validation_data=test_generator,
                               validation_steps=500,
                               verbose=1)

#%%
model_fn ='unet_liverseg_model1_noaugmentation_090117_cont_train_after_epoch10'
model.save(model_fn)

#%%
# visualize the prediction 

y_pred = model.predict(x_test,batch_size=32)
#%%
for s in range(0,1855,50):
    imshow(x_test[s,:,:,0],y_pred[s,:,:,0],y_test[s,:,:,0])