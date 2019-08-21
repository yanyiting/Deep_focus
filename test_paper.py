#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 03:06:30 2019
test .h5 file
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
f = h5py.File('/cptjack/totem/yanyiting/outoffocus2017_patches5Classification.h5')

data=f['/X'].value
label = f['/Y'].value
label_clear=[0.0, 5.0, 6.0]
for k in range(len(label)):
    if label[k] in label_clear:
        label[k]=0
    else:
        label[k]=1

        
#train_images, valid_images, train_labels, valid_labels = train_test_split(data, label, test_size=0.2, random_state=0)
train_images = [imresize(image, (72,72)) for image in data]
train_images = np.array(train_images)
train_images = (train_images/255).astype(np.float32)
Y_train = to_categorical(label,num_classes = np.unique(label).shape[0])

batch_size_for_generators = 32
train_datagen = DataGenerator(featurewise_center=True,featurewise_std_normalization=True)
train_gen = train_datagen.flow(train_images,Y_train,batch_size=batch_size_for_generators)
train_steps = train_images.shape[0]//batch_size_for_generators
train_gen.reset()

model = load_model('/cptjack/totem/yanyiting/deepfocus/new_data/DeepFocus/model/Xception_9/Xception.hdf5')
test_loss, test_accuracy = model.evaluate_generator(train_gen,verbose=1,steps=train_steps)
predictions = model.predict_generator(train_gen,verbose=1,steps=train_steps)
a = np.argmax(predictions, axis=1)


