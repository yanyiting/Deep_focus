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
from scipy.misc import imresize



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Xception_model = load_model('./Resnet50_89/ResNet50.hdf5')
workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/deepfocus/FocusPath/DatabaseInfo.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/deepfocus/FocusPath/DatabaseInfo.xlsx")
label_name = workbook[['Name','Slice #']]


def read_directory(directory_name,ResNet50_model):
    
    img_num = os.listdir(directory_name)
    prediction_list = []
    prediction_array = []
    out_img = np.zeros([16,16])
    prediction_pro = []
    for k in range(len(img_num)):
        img = cv2.imread(directory_name+'/'+img_num[k])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        patch_img=[]
        for i in range(img.shape[1]//64):
            for j in range(img.shape[0]//64):
                #patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
                patch= img[64*i:64*(i+1),64*j:64*(j+1)]
#                patch = img[i/2+32:(i+1)/2-32, j/2+32:(j+1)/2-32]
#                patch = img[(x-32)/2:(x-32)/2+32,(y-32)/2:(y-32/2+32)]
#                patch = img[64*i:64*(i+1),64*j:64*(j+1)]
                patch_img.append(patch)
        patch_img = [imresize(image, (72, 72)) for image in patch_img] ### Reshape to (299, 299, 3) ###
        patch_img = np.array(patch_img)
        patch_img = (patch_img/255).astype(np.float32)
        #patch_img = (patch_img/255)
        Resnet_preds = Xception_model.predict_classes(patch_img)
        if Xception_preds != label_name:
            out_img[i,j]=255
            io.imsave(map_dir+'/'+name+'_1.png',out_img1)
            cv2.imwrite(map_path,out_img)
            
            
        
        Resnet_pro = Resnet_model.predict(patch_img)
        prediction= Counter(Resnet_preds).most_common(1)[0][0]
        print(prediction)
        hunxiao = 0
        if (1 in Xception_preds)and (0 in Xception_preds):
            hunxiao=1
            print(prediction)

        prediction_list.append((img_num[k].split(".")[0],prediction,hunxiao))
        prediction_array.append(Xception_preds)
        prediction_pro.append(Xception_pro)
    return prediction_list,prediction_array,prediction_pro
        
   
directory_name = '/cptjack/totem/yanyiting/deepfocus/FocusPath/FocusPath'
prediction_list, prediction_array,prediction_pro = read_directory(directory_name,Resnet_model)
np.save('prediction_list_01.npy', prediction_list)                    
np.save('prediction_array_01.npy', prediction_array) 
np.save('prediction_pro_01.npy',prediction_pro)                  

del out_img,pre_img
gc.collect()   


    #plt.savefig('../deepfocus/confusion_matrix.png',format = 'png')
    
   

