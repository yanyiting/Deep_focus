#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 03:06:30 2019
train = origin64
test = origin64

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

os.environ['CUDA_VISIBLE_DEVICES']='0'
#def set_config():
#    os.environ['CUDA_VISIBLE_DEVICES']='1'
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#    config = tf.ConfigProto(gpu_options=gpu_options)
#    config = tf.Session(config = config)
#    return config

#cfg = set_config()
#with tf.Session(config=cfg) as sess:

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
n_epochs = 1000
sgd = optimizers.SGD(lr=LearningRate,decay=LearningRate/n_epochs,momentum = 0.9,nesterov = True)
top_model.compile(optimizer = sgd,loss = 'categorical_crossentropy',metrics=['accuracy'])
trainable_params = int(np.sum([K.count_params(p)for p in set(top_model.trainable_weights)]))
non_trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.non_trainable_weights)]))
print("model Stats")
print("="*30)
print("Total Parameters:{:,}".format((trainable_params+non_trainable_params)))
print("Non-Trainable Parameters:{:,}".format(non_trainable_params))
print("Trainable Parameters:{:,}\n".format(trainable_params))
train_folders = '/cptjack/totem/yanyiting/deepfocus/FocusPath/train64/'
validation_folders = '/cptjack/totem/yanyiting/deepfocus/FocusPath/val64/'

img_width,img_height = 72,72
#x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
#x_validation = x_validation(x_test.shape[0],img_rows,img_cols,1)

        
        #clear:0
        #nonclear:1
batch_size_for_generators = 32
train_datagen = DataGenerator(rescale=1./255, rotation_range=178,horizontal_flip=True,vertical_flip=True,shear_range=0.6,stain_transformation = True)
train_generator = train_datagen.flow_from_directory(train_folders,batch_size=batch_size_for_generators,target_size=(img_width,img_height))
validation_generator = DataGenerator(rescale=1./255)
validation_generator = validation_generator.flow_from_directory(validation_folders,batch_size = batch_size_for_generators,target_size=(img_width,img_height))

nb_train_samples = sum([len(files)for root,dirs,files in os.walk(train_folders)])
nb_validation_samples = sum([len(files)for root,dirs,files in os.walk(validation_folders)])



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
        
                
        
            
file_path = "Xception.hdf5"
callbacks_s = get_callbacks(file_path,top_model,9,patience=5)
train_steps = nb_train_samples//batch_size_for_generators
valid_steps = nb_validation_samples//batch_size_for_generators
top_model.fit_generator(generator=train_generator,epochs=n_epochs,steps_per_epoch=train_steps,validation_data=validation_generator,
                        validation_steps = valid_steps, callbacks=callbacks_s, verbose=1)
