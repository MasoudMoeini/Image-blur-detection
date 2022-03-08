import os
import numpy as np
from numpy import asarray
import cv2


def load_images_from_folder(path):
    list_image_names=[]
    images = []
    for item in os.listdir(path): # we iterate over the list of all file names in folder
        list_image_names.append(item)
        
    for img in list_image_names:
        img = cv2.imread(os.path.join(path,img))
        data = asarray(img).astype(np.float32)
        images.append(data)
        
    print(f" {len(images)} from {path} successfully uploaded")      
    return images

import os
import numpy as np
from numpy import asarray
import cv2


def load_images_from_folder(path):
    list_image_names=[]
    images = []
    for item in os.listdir(path): # we iterate over the list of all file names in folder
        list_image_names.append(item)
        
    for img in list_image_names:
        img = cv2.imread(os.path.join(path,img))
        #data = asarray(img).astype(np.float32)
        data = np.array(img, dtype = float) / 255.0
        reshaped = data.reshape([-1, 3, 244, 244])
        reshaped = reshaped.transpose([0, 2, 3, 1])
        images.append(reshaped)
        
    print(f" {len(images)} from {path} successfully uploaded")      
    return images