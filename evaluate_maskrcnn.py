import os
import numpy as np
import json
from detectron2.structures import BoxMode
import cv2
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
from dt2trainer import DefaultClutterizedTrainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, help="Data directory path", default="./Individual/")
parser.add_argument("--maxIter", type=int, help="Total number of iterations to train", default=100000)
parser.add_argument("--batchSize", type=int, help="Size of Batch", default=16)
parser.add_argument("--inputSize", type=int, help="Resolution for the background image (and therefore the cluttered image", default=512)
parser.add_argument("--objectSize", type=int, help="Resolution for each object image", default = 384)
parser.add_argument("--learningRate", type=float, help="Set the learning rate", default=0.0025 )
parser.add_argument("--modelYAML", type=str, help="Set the model YAML [Visit: https://github.com/facebookresearch/detectron2/tree/master/configs/COCO-InstanceSegmentation]", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
parser.add_argument("--numWorkers", type=int, help="Set the number of workers for the Detectron2 dataloader", default=2)
parser.add_argument("--saveAt", type=int, help="Iteration number multiples at which to save weights, -1 means weights aren't saved", default=500)
parser.add_argument("--weightsLoc", type=str, help="Location of the weights file")

args = parser.parse_args()

def get_class_idx_mappings(train_dataset_dir):
    i_to_c = []
    for object_name in os.listdir(train_dataset_dir):
        if os.path.isdir(os.path.join(train_dataset_dir, object_name)):
            i_to_c.append(object_name)
    i_to_c.remove('background')
    i_to_c.sort()
    c_to_i = {}
    for i in range(len(i_to_c)):
        c_to_i[i_to_c[i]] = i
    return i_to_c, c_to_i
    
def get_all_axis(polygon, axis):
    values = []
    for i in range(len(polygon)):
        values.append(polygon[i][axis])
    return values
    

def get_dataset_dict():
    test_dir = "./TestDataset/"
    train_dir = "./Individual/"
    _, c_to_i = get_class_idx_mappings(train_dir)
    dataset_dicts = []
    idx = 0
    for img_name in os.listdir(test_dir):
        if img_name.endswith('.JPG') or img_name.endswith('jpg'):
            json_name = img_name.split('.')[0]+'.json'
            with open(os.path.join(test_dir, json_name)) as f:
                json_data = json.load(f)
            
            record = {}
                
            filename = os.path.join(test_dir, img_name)
            height, width = cv2.imread(filename).shape[:2]
                
            record["file_name"] = filename
            record["image_id"] = idx
            idx += 1
            record["height"] = height
            record["width"] = width
            objs = []
            for i in range(len(json_data['shapes'])):
                shape_str = json_data['shapes'][i]
                polygon_str = shape_str['points']
                label_str = shape_str['label']
                #print(polygon_str)
                #print(label_str)
                polygon = []
                for i in range(len(polygon_str)):
                    polygon.append(list(map(int, polygon_str[i])))
                all_x = get_all_axis(polygon, 0)
                all_y = get_all_axis(polygon, 1)
                polygon = [p for x in polygon for p in x]  
                obj = {
                    "bbox":[np.min(all_x), np.min(all_y), np.max(all_x), np.max(all_y)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation":[polygon],
                    "category_id": c_to_i[label_str],
                    "is_crowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

#from detectron2.data import DatasetCatalog, MetadataCatalog
#for d in ["train", "val"]:
#    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
#balloon_metadata = MetadataCatalog.get("balloon_train")

from detectron2.data import DatasetCatalog, MetadataCatalog

i_to_c, c_to_i = get_class_idx_mappings("./Individual/")
print(i_to_c)
print(c_to_i)
DatasetCatalog.register("test_clutterized", get_dataset_dict)
MetadataCatalog.get("test_clutterized").set(thing_classes=i_to_c)
clutterized_metadata = MetadataCatalog.get("test_clutterized")

if not os.path.exists('./sample_test_images/'):
    os.mkdir('./sample_test_images/')

dataset_dicts = get_dataset_dict()
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=clutterized_metadata, scale=0.4)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite('./sample_test_images/'+ d["file_name"].split("/")[-1], vis.get_image()[:, :, ::-1])

if not os.path.exists('./test_output'):
    os.mkdir('./test_output')
selectedModel = args.modelYAML
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(selectedModel))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(selectedModel)
cfg.MODEL.NAME = selectedModel.split('/')[-1].split('.')[0]
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# let training initialize from model zoo
cfg.DATALOADER.NUM_WORKERS = args.numWorkers
cfg.DATALOADER.DATASET_LOCATION = args.dataDir
cfg.DATALOADER.INPUTSIZE = args.inputSize
cfg.DATALOADER.OBJECTSIZE = args.objectSize
cfg.SOLVER.IMS_PER_BATCH = args.batchSize
cfg.SOLVER.BASE_LR = args.learningRate  # pick a good LR 
cfg.MODEL.SAVEAT = args.saveAt
cfg.INPUT.MASK_FORMAT = "poly"

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # only has one class (ballon)

cfg.OUTPUT_DIR = './output_test'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("Declaring Trainer")
trainer = DefaultClutterizedTrainer(cfg)
trainer.load_state_dict(torch.load(args.weightsLoc, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
evaluator = COCOEvaluator("test_clutterized", cfg, True, output_dir='./test_output')
val_loader = build_detection_test_loader(cfg, "test_clutterized")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
