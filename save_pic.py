import numpy as np
import imageio
import h5py
from skimage import img_as_ubyte
import os

f = h5py.File('/cptjack/totem/yanyiting/outoffocus2017_patches5Classification.h5')
data=f['/X'].value
label = f['/Y'].value

for i in range(data.shape[0]):
    sr = data[i]
    
    img_uint8 = img_as_ubyte(sr)
    img_uint8 = np.clip(img_uint8, 0, 255)
    img_path='/cptjack/totem/yanyiting/deepfocus/picture2/'+ str(int(label[i]))
    if not os.path.exists(img_path):os.makedirs(img_path)
    imageio.imsave(img_path+'/' +str(i)+'.png',img_uint8)


