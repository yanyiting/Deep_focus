#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 02:28:36 2019

@author: root
"""

import cv2
import os
import numpy as np

img_dir = '/cptjack/totem/yanyiting/deepfocus/FocusPath/FocusPath/'
img_names = sorted(os.listdir(img_dir))
for i in range(len(img_names)):
    img = cv2.imread(img_dir+img_names[i], 0)
    new_img = np.array([img,img,img])
    new_img = new_img.transpose((1,2,0))
    cv2.imwrite('/cptjack/totem/yanyiting/deepfocus/FocusPath/gray_picture/'+img_names[i], new_img)

