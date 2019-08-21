#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 03:06:30 2019

@author: root
"""
import pandas as pd
import numpy as np
import argparse
import datetime
import random
import keras
import glob
import time
import sys
import os

from keras.models import *
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,MaxPooling2D,Flatten,BatchNormalization
from keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import classification_report,confusion_matrix
from keras.applications.xception import Xception
from keras.initializers import orthogonal
from keras.utils import to_categorical
from generators import DataGenerator
from scipy.misc import imresize
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io
from keras.callbacks import CSVLogger,Callback,EarlyStopping
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.utils import multi_gpu_model
import time
from PIL import Image
import h5py
from sklearn.model_selection import StratifiedKFold
from scipy import misc
from sklearn.model_selection import train_test_split
import h5py
import gc
import tensorflow as tf


#def set_config():
#    os.environ['CUDA_VISIBLE_DEVICES']='1'
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#    config = tf.ConfigProto(gpu_options=gpu_options)
#    config = tf.Session(config = config)
#    return config

#cfg = set_config()
#with tf.Session(config=cfg) as sess:
    
os.environ['CUDA_VISIBLE_DEVICES']='3'
f = h5py.File('/cptjack/totem/yanyiting/backup/outoffocus2017_patches5Classification.h5')
data=f['/X'].value
#print(data)
label = f['/Y'].value
label_clear=[0.0, 5.0, 6.0]
for k in range(len(label)):
    if label[k] in label_clear:
        label[k]=0
    else:
        label[k]=1
'''er ba fen'''
train_images = [] 
valid_images= []
train_labels =[] 
valid_labels = []
data_1 = [imresize(image, (72,72)) for image in data]
for i in range(len(data_1)):
    img_bgr_data = data_1[i]
    #>=94 and <=222
    region_flag_r = ((abs(img_bgr_data[:,:,0]-160)<=62) & (abs(img_bgr_data[:,:,1]-103)<=61)&(abs(img_bgr_data[:,:,2]-168)<=46))
    nulei_flag_r = round(np.sum(region_flag_r)/(64*64),4)
    region_flag_b = ((abs(img_bgr_data[:,0,:]-206)<=236)&(abs(img_bgr_data[:,1,:]-196)<=240)&(abs(img_bgr_data[:,2,:]-190)<=241))
    nulei_flag_b = round(np.sum(region_flag_b)/(64*64),4)
#    region_flag_y = ((abs(img_bgr_data[0,:,:]-144)<=72)&(abs(img_bgr_data[1,:,:]-153)<=81)&(abs(img_bgr_data[2,:,:]-175)<=71))
#    nulei_flag_y = round(np.sum(region_flag_y)/(64*64),4)
    
    if (nulei_flag_r>0.038)or(nulei_flag_b>0.0503):
        train_images.append(img_bgr_data)
        train_labels.append(label[i])
    else:
        valid_images.append(img_bgr_data)
        valid_labels.append(label[i])
        
train_images, valid_images, train_labels, valid_labels = train_test_split(data, label, test_size=0.2, random_state=0)
#train_images = [imresize(image, (72,72)) for image in train_images]
train_images = np.array(train_images)
train_images = (train_images/255).astype(np.float32)
Y_train = to_categorical(train_labels,num_classes = np.unique(train_labels).shape[0])
#valid_images = [imresize(image, (72,72)) for image in valid_images]
valid_images = np.array(valid_images)
valid_images = (valid_images/255).astype(np.float32)
Y_valid = to_categorical(valid_labels,num_classes = np.unique(valid_labels).shape[0])

#data = [np.array(Image.fromarray(image).resize(72,72))for image in data]
'''si zhe jiao cha yan zheng 

all_scores = []
stratified_folder=StratifiedKFold(n_splits = 4, random_state=0, shuffle=False)
for i,(train_index, test_index) in enumerate(stratified_folder.split(data, label)):
    #print('train_x',train_index)
    #print('test_index',test_index)
    #print('i:',i)
    train_images = data[train_index]
    train_labels = label[train_index]
    train_images = [imresize(image, (72,72)) for image in train_images]
    train_images = np.array(train_images)
    train_images = (train_images/255).astype(np.float32)
    Y_train = to_categorical(train_labels,num_classes = np.unique(train_labels).shape[0])
    
    
    valid_images = data[test_index]
    valid_labels = label[test_index]
    valid_images = [imresize(image, (72,72)) for image in valid_images]
    valid_images = np.array(valid_images)
    valid_images = (valid_images/255).astype(np.float32)
    Y_valid = to_categorical(valid_labels,num_classes = np.unique(valid_labels).shape[0])
    '''

#base = Xception(weights = 'imagenet', include_top=False, input_shape=(72,72,3))
#base.save('Xception(72).h5')   
base = load_model('Xception(72).h5')
top_model = Sequential()
top_model.add(base)
top_model.add(MaxPooling2D(pool_size=(2,2)))
top_model.add(Flatten())
top_model.add(Dropout(0.5))
top_model.add(Dense(128,activation = 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2,activation='softmax',kernel_initializer=orthogonal()))
top_model.summary()
for layer in top_model.layers:
    layer.trainable = True
    
LearningRate = 0.001
decay = 0.0001
n_epochs = 50
sgd = optimizers.SGD(lr=LearningRate,decay=LearningRate/n_epochs,momentum = 0.9,nesterov = True)
top_model.compile(optimizer = sgd,loss = 'categorical_crossentropy',metrics=['accuracy'])
trainable_params = int(np.sum([K.count_params(p)for p in set(top_model.trainable_weights)]))
non_trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.non_trainable_weights)]))
print("model Stats")
print("="*30)
print("Total Parameters:{:,}".format((trainable_params+non_trainable_params)))
print("Non-Trainable Parameters:{:,}".format(non_trainable_params))
print("Trainable Parameters:{:,}\n".format(trainable_params))
    
        
        #clear:0
        #nonclear:1
batch_size_for_generators = 32
train_datagen = DataGenerator(rotation_range=178,horizontal_flip=True,vertical_flip=True,shear_range=0.6,stain_transformation = True)
train_gen = train_datagen.flow(train_images,Y_train,batch_size=batch_size_for_generators)
valid_datagen = DataGenerator()
valid_gen = valid_datagen.flow(valid_images,Y_valid,batch_size = batch_size_for_generators)
start = time.time()
class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_loss',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)
        
def get_callbacks(filepath,model,j,patience=8):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, './Xception_'+str(j)+'/'+filepath)
    file_dir = './Xception_'+str(j)+'/'+'log/'+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./Xception_'+str(j)+'/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))  +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]
        
        
            
import time
file_path = "Xception.hdf5"
callbacks_s = get_callbacks(file_path,top_model,8,patience=5)
train_steps = train_images.shape[0]//batch_size_for_generators
valid_steps = valid_images.shape[0]//batch_size_for_generators
top_model.fit_generator(generator=train_gen,epochs=n_epochs,steps_per_epoch=train_steps,validation_data=valid_gen,
                        validation_steps = valid_steps, callbacks=callbacks_s, verbose=1)
#top_model=load_model('/cptjack/totem/yanyiting/deepfocus/Xception_'+str(i)+'/Xception.hdf5')
#val_loss, val_metrics=top_model.evaluate(valid_images, Y_valid, batch_size =32)
#all_scores.append((val_loss, val_metrics))
#    del train_images
#    del valid_images
#    del top_model
#    gc.collect()
##    
#    
#    
#    
#    
#    
#    
#    
#    
#all_scores = np.array(all_scores)
#np.save('all_scores.npy',all_scores)
