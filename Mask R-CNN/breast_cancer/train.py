import warnings
warnings.filterwarnings("ignore")
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import mrcnn.model as MaskRCNN
from mrcnn.model import log
from mrcnn import model as  modellib, utils

class CustomConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+2
    STEPS_PER_EPOCH = 51
    DETECTION_MIN_CONFIDENCE = 0.1
    
'''load_custom
   1)add classes 
   2)check type of dataset val or train
   3)load the json file -> store each value in value of annotation file in a list->include only the image files annotaions which includes region attributes
   4)extract the shape and class names attributes -> replace the class names with integer
   5)acess the image and read its metadata eg: width , height -> add the image with its corresponding attributes in dictionary'''
  
class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class("object", 1, "benign")
        self.add_class("object", 2, "malignant")
        assert subset in ["train", "val"]#checks whther the passed paramenter is train or val when the method is called
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open(""))#reads and loads json file
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]#filters the annotaions which includes region data
        for a in annotations:
            polygons = [s['shape_attributes'] for s in a['regions']]
            objects = [r['region_attributes']['names'] for r in a['regions']] 
            name_dict = {"benign": 1, "malignant":2}
            num_ids = [name_dict[a] for a in objects]#corresponds the classes stored in objects[m,b,b,m] with numeric value in num_dict  eg: a[m] = num_dict[m] = 2 
            
            image_path = os.path.join(dataset_dir, a[''])
            image = skimage.io.imread(image_path)#reads the image#as_gray = true? as gray converts rgb image to grayscale(3d image(height,width,rgb)(3 represents rgb,4=rgba) to 2d image(h,w))
            height, width = image.shape[:2]#slices the first 2 tuples of the image shape(0 = h,1=w and ignore any additional dmensions)
            
            self.add_image(
                "object", 
                image_id = a['filename'],
                path=image_path,
                width = width,#comes under kwargs(kwargs adds any additional arguments)
                height = height,
                polygons = polygons,
                num_ids = num_ids
            )
            
            '''
            load_mask()
            1) retrieve the saved metadata of the image of a particular given image with help of its image id
            2)check whther the class type in data matches the the defiend classes. if classs not call the parent function and assign zeros to the mask
            3)make and 3d array consisiting of zeros  
            
            np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8):
            
            -----info["height"] and info["width"]: These specify the dimensions of the mask in terms of height and width. These values likely correspond to the dimensions of an image or a specific region of interest.
            -----len(info["polygons"]): The third dimension of the array corresponds to the number of polygons in info["polygons"]. Each polygon might represent a distinct object or region to be labeled in the mask.
            4)converts the x,y coordinates into column and row and the pixels inside are represented by the fprmed polygon and are converted to 1s
            
            '''
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":#if class in object is not equal to in image info
            return super(self.__class__, self).load_mask(image_id)#if class not found then calls the parent class that assigns zeros to the mask
        
        info = self.image_info[image_id]
        num_ids = info['num_ids']#store the unique nums_id
        mask = np.zeros([info["height"],info["width"], len(info["polygons"])],dtype = np.uint8)
        for i,p in enumerate(info["polygons"]): #iterates through each polygon
            rr,cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x']) #converts the x,y coordinates into column and row and the pixels inside are represented by the fprmed polygon
            mask[rr,cc,i] = 1 #converts the ith polygon area with zeros to 1
            num_ids = np.array(num_ids, dtype=np.int32)#num_ids = np.array([1, 2.0, '3'], dtype=np.int32)  # Output: array([1, 2, 3], dtype=int32)

            return mask, num_ids
        
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
def train(model):
    dataset_train = CustomDataset()
    dataset_train.load_custom("")
    dataset_train.prepare()
    
    dataset_val = CustomDataset()
    dataset_val = train.load_custom("")
    dataset_val.prepare()
    
    model.train(dataset_train, dataset_val,
                learning_rate = config.LEARNING_RATE,
                epochs = 10,
                layers = 'head')
config = CustomConfig()
model = modellib.MaskRCNN(mode = 'training', config = config, model_dir = DEFAULT_LOGS_DIR)
weights_path = WEIGHTS_PATHS

if not os.path.exist(weights_path):
    utils.download_trained_weights(weights_path)
    model.load_weights(weights_path, by_name = True, excludes=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    train(model)
    
TEST_MODE = "inference"
ROOT_DIR = ""
def get_ax(rows = 1, cols=1, size = 16):
    
    _, ax = plt.subplots(rows, cols, figsize = (size*cols, size*rows))
    return ax
CUSTOM_DIR = " "
dataset = CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

model = modellib.MaskRCNN(mode = "inference", model_dir = MODEL_DIR, config = config)

weights_path = WEIGHTS_PATH
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name = True)

#run detection
image_id = random.lchoice(dataset.image_ids)
print("image id is :", image_id)

image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(datset, config, image_id, use_mini_mask = False)
results = model.detect([image], verbose=1)
 
x = get_ax(1)
r = results[0]
ac = plt.gca()

visualize.display_instances(image, r['rois'], r['mask'], r['class_ids'], datset.class_names, r['scores'], ax =ax, title = "Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox) 
log("gt_mask", gt_mask) 