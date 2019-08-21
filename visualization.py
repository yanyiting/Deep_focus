#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 03:33:04 2019

@author: root
"""

import scipy.io as scio
from skimage import io
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import glob
from keras.models import *
import time
import openslide as opsl
from collections import Counter


Xception_model = load_model('./Xception_9/Xception.hdf5')


def read_directory(directory_name,Xception_model):
    img_num = os.listdir(directory_name)
    prediction_list = []
    for k in range(len(img_num)):
        img = cv2.imread(directory_name+'/'+img_num[k])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img/255
        patch_img=[]
        for i in range(img.shape[1]//72):
            for j in range(img.shape[0]//72):
                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
                #patch = img[64*i:64*(i+1),64*j:64*(j+1)]
                patch_img.append(patch)
        patch_img = np.array(patch_img)
        Xception_preds = Xception_model.predict_classes(patch_img)
        prediction= Counter(Xception_preds).most_common(1)[0][0]
        print(prediction)
        prediction_list.append((img_num[k].split(".")[0],prediction))
    return prediction_list
        
   
directory_name = '/cptjack/totem/yanyiting/deepfocus/FocusPath/FocusPath'
prediction_list = read_directory(directory_name,Xception_model)
np.save('prediction_list.npy', prediction_list)                    
                    

                        