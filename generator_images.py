import os
import numpy as np
from numpy import asarray
import cv2


def load_images_from_folder(path):
    list_image_names=[]
    images = np.empty((1, 224, 224, 3), np.float32)
    for item in os.listdir(path): # we iterate over the list of all file names in folder
        list_image_names.append(item)
        
    for img in list_image_names:
        img = cv2.imread(os.path.join(path,img))
        #data = asarray(img).astype(np.float32)
        data = np.array(img, dtype = float) / 255.0
        reshaped = data.reshape([-1, 3, 244, 244])
        reshaped = reshaped.transpose([0, 2, 3, 1])
        images=np.append(images, reshaped, axis=0)
        #images.append(reshaped)
        
    print(f" {len(images)} from {path} successfully uploaded")      
    return images

