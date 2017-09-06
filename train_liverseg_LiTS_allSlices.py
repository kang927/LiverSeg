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

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize,imsave
from keras import initializers 

import nibabel as nib # for reading nifTi data file 
from unet import UNet, dice_coef_loss, dice_coef

# for regular expression
import re


PAD_SIZE = 48

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


def listdir_fullpath(dn):
    return [os.path.join(dn,fn) for fn in os.listdir(dn)]




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


def listdir_fullpath(dn):
    return [os.path.join(dn,fn) for fn in os.listdir(dn)]




def batch_read(data_list,batch_size,ii):
    """ 
    load up batch of data and prepare it for training 
    Argument:
        data_list: tuple of img and seg file list
        batch_size: number of batch data to load
        ii: index indicates starts reading from iith file in data_list
    
    """
    ct = 0
    Nslice = len(data_list[0])
    if ii+batch_size >= Nslice:
        index_end = Nslice
    else:
        index_end = ii+batch_size
    
    for s in range(ii,index_end,1):
        if s==ii:
            tmp= np.load(data_list[0][s])
            nx = tmp.shape[0]
            ny = tmp.shape[1]
            img = np.zeros( (batch_size,nx,ny,1) )
            seg = np.zeros( (batch_size,nx,ny,1) )
            img[ct,:,:,0] = tmp
            seg[ct,:,:,0] = np.load(data_list[1][s])
        
        else:
            img[ct,:,:,0] = np.load(data_list[0][s])
            seg[ct,:,:,0] = np.load(data_list[1][s])
        ct = ct+1
    # repackage the shape to make it fit keras training data 
    #print("the img size is" + str(img.shape) + "\n")
    return img, seg, s+1



def batch_generator(data_list,
                    batch_size,
                    seed=None):
    """ 
    generator that iterate through batch of volume data and generate
    augmented 2D images for training
    
    Argument:
        img_seg_generator: keras ImageDataGenerator object for images and segmentations 
        data_list: a list contain the directory of data volume in .npy format
        batch_vol: how many volume data to load at each iteration
        batch_size: number of images to generate per iteration
    """
    Nslices = len(data_list[0]) # number of data slices to loop through
    ii=0
    
    while 1:
        #print("using data slice #" + str(ii) + "\n")
        # load batches of img and segmentation        
        x_train, y_train, ii = batch_read(data_list,batch_size,ii)
       # reset if we reach the end of all files
        if ii%Nslices ==0:
            #print("reseting ii to 0\n")
            ii=0 # if we loop through all data vol, repeat again
        
        x_train = np.pad(x_train,((0,0),(PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
        y_train = np.pad(y_train,((0,0),(PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
        yield x_train, y_train


#def split_data(data_list,split_percent,seed=None):
#    """
#    split the list of images and segmentation files into training and testing 
#    Argument:
#        data_list: pair of images, segmentation directory list
#        split_percent: what percent of data is used for testing
#    Return:
#        train_list: tuple of (images,segmentation) file list
#        test_list: tuple of (images,segmentation) file list
#    
#    """
#    train_img_list = list()
#    train_seg_list = list()
#    test_img_list = list()
#    test_seg_list = list()
    # loop through each file in the list and get their image volume and slice number


def check_img_seg_pair(img_list,seg_list):
    """ make sure the pair in the list match each other by checking image and slice number """
    
    Nl = len(img_list)
    p = re.compile('\d+')
    for ii in range(Nl):
        tmp1 = p.findall(img_list[ii])
        tmp2 = p.findall(seg_list[ii])
        if np.any([i!=j for i,j in zip(tmp1,tmp2)]):
            print('img_list and seg_list dont match \n' + 
                             'img_list: ' + img_list[ii] + '\n' +
                                'seg_list: ' + seg_list[ii] + '\n'
                                )




train_img_dir = 'D:/liverseg_training/training_slice_sagittal/images/'
train_seg_dir = 'D:/liverseg_training/training_slice_sagittal/segmentations/'
list_img = listdir_fullpath(train_img_dir)
list_seg = listdir_fullpath(train_seg_dir)

Nslices = len(list_img)
# spliting into training and test set 
test_end = 5799 # using 20 cases as the testing
test_img = list_img[0:test_end]
test_seg = list_seg[0:test_end]
train_img = list_img[test_end:]
train_seg = list_seg[test_end:]


# sainity check for match images/segmentation pairs before training
check_img_seg_pair(test_img,test_seg)
check_img_seg_pair(train_img,train_seg)

# obtaining the testing set 
test_list = (test_img,test_seg)
train_list = (train_img,train_seg)





train_img_dir = 'D:/liverseg_training/training_slice_sagittal/images/'
train_seg_dir = 'D:/liverseg_training/training_slice_sagittal/segmentations/'
list_img = listdir_fullpath(train_img_dir)
list_seg = listdir_fullpath(train_seg_dir)