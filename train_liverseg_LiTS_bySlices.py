
# coding: utf-8

# In[1]:


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

# use for natural order sorting
from natsort import natsorted

# random shuffle input images for better training
from random import shuffle
import random

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


# In[2]:


#%% setup utility functions 
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


# In[3]:


def listdir_fullpath(dn):
    
    tmp = [os.path.join(dn,fn) for fn in os.listdir(dn)]
    # sort it at natural order (i.e. 1 2 3 4 ... 10 instead of 1 10 100 etc )
    lst_dir = natsorted(tmp) 
    return lst_dir


# In[4]:


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
    
    # take care of the case that the next batch of data run over the end of all slices
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


# In[5]:


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
        
        #x_train = np.pad(x_train,((0,0),(PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
        #y_train = np.pad(y_train,((0,0),(PAD_SIZE,PAD_SIZE),(PAD_SIZE,PAD_SIZE),(0,0)),mode='constant',constant_values=0)
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


# In[6]:


train_img_dir = '/data/train_axial/images/'
train_seg_dir = '/data/train_axial/segmentations/'
list_img = listdir_fullpath(train_img_dir)
list_seg = listdir_fullpath(train_seg_dir)

Nslices = len(list_img)
# spliting into training and test set 
test_end = 2000#5799 # using 20 cases as the testing
test_img = list_img[0:test_end]
test_seg = list_seg[0:test_end]
train_img = list_img[test_end:]
train_seg = list_seg[test_end:]


# sainity check for match images/segmentation pairs before training
check_img_seg_pair(test_img,test_seg)
check_img_seg_pair(train_img,train_seg)

# obtaining the testing set 
test_list = (test_img,test_seg)

# for the training set we want to randomly shuffle the input data for better training
tmp = list(zip(train_img,train_seg))
shuffle(tmp)
train_img,train_seg = zip(*tmp)
train_list = (train_img,train_seg)


# In[7]:


# testing for the generator
#ct = 0
#train_generator = batch_generator(train_list, 500)
#test_generator = batch_generator(test_list,500)
#for x_test, y_test in train_generator:
#    print x_test.shape
#    imshow(x_test[5,:,:,0],y_test[5,:,:,0])
#    ct = ct+1
#    if ct>10:
#        break
#tmp = next(test_generator)
#print(hasattr(tmp, '__len__'))


# In[8]:


#%% initialize the model and training parameters
batch_size = 4
x_sample, y_sample,ii = batch_read(test_list,2,0)
n_train = len(train_img)
n_test = len(test_img)
print("using "+ str(n_train)+" for training and using "+ str(n_test)+" for testing")

nx = x_sample.shape[1]
ny = x_sample.shape[2]
n_channels = x_sample.shape[3]

# make the training and test data generator
train_generator = batch_generator(train_list, batch_size)
test_generator = batch_generator(test_list, batch_size)


# In[9]:


# setup the model 
get_ipython().magic(u'run unet_noisy.py')


# In[10]:


stddev=0.01
model = UNet( input_shape=(nx,ny,n_channels),stddev=stddev)


# In[11]:


# setup learning parameters and metric for optimization 
model.compile(optimizer=Adam(lr=1e-3), loss=jacc_dist, metrics=[dice_coef])


# In[ ]:


#%% start training
num_epochs=50
n_per_epoch = np.round(n_train/batch_size)
n_test_steps = np.round(n_test/batch_size)
hist1 = model.fit_generator(train_generator,steps_per_epoch=n_per_epoch,epochs=num_epochs,
                               validation_data=test_generator,
                               validation_steps=n_test,
                               verbose=1)



# In[ ]:


#%%
model_fn ='/src/LiverSeg/unet_liverseg_model_kernel6'
model.save(model_fn)


# In[ ]:




