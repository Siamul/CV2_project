import cv2
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from numpy import asarray
import shutil

data_dir = '/afs/crc.nd.edu/user/s/skhan22/cv2-clutterized-dataloader/Individual/'
save_dir = '/afs/crc.nd.edu/user/s/skhan22/cv2-clutterized-dataloader/Dataset/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for img_dir in os.listdir(data_dir):
    for img_name in tqdm(os.listdir(data_dir+img_dir)):
        if img_name.endswith('.JPG'):
            json_name = img_name.split('.')[0]+'.json'
            with open(data_dir+img_dir+'/'+json_name) as f:
                json_data = json.load(f)
            mask_polygon_str = json_data['shapes'][0]['points']
            #print(mask_polygon_str)
            mask_polygon = []
            for i in range(len(mask_polygon_str)):
                mask_polygon.append(list(map(int, mask_polygon_str[i])))
            mask_polygon = np.array(mask_polygon, dtype=np.int32)
            image = asarray(cv2.imread(data_dir+img_dir+'/'+img_name))
            mask = np.int32(np.zeros((int(json_data['imageHeight']), int(json_data['imageWidth']))))
            cv2.fillPoly(mask, [mask_polygon], color=(255))
            mask_binary = mask / 255
            mask_binary = mask_binary.astype(int)
            #print(image.shape)
            #print(mask_binary.shape)
            mask_binary = mask_binary.reshape(mask_binary.shape[0], mask_binary.shape[1], 1)
            mask_binary_3d = np.concatenate((mask_binary, mask_binary, mask_binary), axis=2)
            #print(mask_binary_3d.shape)
            masked_image = np.multiply(image, mask_binary_3d)
            if not os.path.exists(save_dir+img_dir):
                os.mkdir(save_dir+img_dir)
            cv2.imwrite(save_dir+img_dir+'/'+img_name, masked_image)
            #shutil.copy(data_dir+img_dir+'/'+json_name, save_dir+img_dir+'/'+json_name)
            
              
