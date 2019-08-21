#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import h5py
import numpy as np
from scipy.misc import imsave
from skimage import transform, io, img_as_ubyte
import tensorflow as tf
#import imageio.imwrite
def load_dataset():
    train_dataset = h5py.File('/cptjack/totem/yanyiting/outoffocus2017_patches5Classification.h5')
   # train_set_x_orig = np.array(train_dataset["train_set_x"][:])
   # train_set_x_orig = np.array(train_dataset["train_set_y"][:])
    
def processing():
    X_train_orig,Y_train_orig,classes = load_dataset()
    m = len(X_train_orig)
    Y_train_t = Y_train_orig.T
    
    for i in range(m):
        name = 'images/train/'+str(i)+'-['+str(np.squeeze(Y_train_t[i]))+'].jpg'
        imsave(name,transform.rescale(X_train_orig[i].reshape(64,64,3),10,mode = 'constant'))
def reading():
    image = cv2.imread('imgaes/train/16-[1].jpg',cv2.IMREAD_UNCHANGED)
    print(image.shape)
    cv2.imshow('input_image',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('input_image',transform.rescale(image,0.5,mode = 'constant'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    processing()


f = h5py.File('/cptjack/totem/yanyiting/outoffocus2017_patches5Classification.h5')
data=f['/X'].value
label = f['/Y'].value
data[100:200,:,:,:]
#print(data[0])
# len(data) = 204000  -----total
# len(data[0]) = 64 ---- x
#len(data[0][0]) = 64  ---- y
#len(data[0][0][0])=3  ---- rgb


sr = data[100:200,:,:,:] *255.0

#sr = tf.cast(sr, tf.int32)
#sr = tf.maximum(sr, 0)
#sr = tf.minimum(sr, 255)
#sr=tf.cast(sr, tf.uint8)
#sr = tf.cast(sr, tf.int32)
sr = tf.maximum(sr, 0)
sr = tf.minimum(sr, 255)
sr=tf.cast(sr, tf.uint8)
img_uint8 = img_as_ubyte(sr)
img_uint8 = np.clip(img_uint8, 0, 255)
img_uint8[0,:,:,:].shape
for i in range(sr.shape[0]):
    imwrite('/cptjack/totem/yanyiting/picture1/'+str(i)+'.png', img_uint8[i,:,:,:])     
            
    