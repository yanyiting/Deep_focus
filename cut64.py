#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:05:24 2019

@author: root
"""

import pandas as pd
import numpy as np
import os
import cv2
import xlrd

workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/deepfocus/FocusPath/DatabaseInfo.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/deepfocus/FocusPath/DatabaseInfo.xlsx")
label_name = workbook[['Name','Slice #']]
label_name['Name'][0]
directory_name = '/cptjack/totem/yanyiting/deepfocus/FocusPath/gray_picture'
img_num =sorted( os.listdir(directory_name))

focus = [7,8,9,10]

for k in range(len(img_num)):
        img = cv2.imread(directory_name+'/'+img_num[k])
        if k<672:         
            if label_name['Slice #'][k] in focus:
                save_dir = '/cptjack/totem/yanyiting/deepfocus/gray_picture/train_gray/focus/'
            else:
                save_dir = '/cptjack/totem/yanyiting/deepfocus/gray_picture/train_gray/unfocus/'
        else:
            if label_name['Slice #'][k] in focus:
                save_dir = '/cptjack/totem/yanyiting/deepfocus/gray_picture/val_gray/focus/'
            else:
                save_dir = '/cptjack/totem/yanyiting/deepfocus/gray_picture/val_gray/unfocus/'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        for i in range(img.shape[1]//64):
            for j in range(img.shape[0]//64):
                #patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
                patch= img[64*i:64*(i+1), 64*j:64*(j+1)]
                cv2.imwrite(save_dir+img_num[k].split('.')[0]+'_'+str(i)+'_'+str(j)+'.png', patch)