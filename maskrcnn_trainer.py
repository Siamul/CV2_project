import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import numpy as np
import cv2
import random

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import logging
import time
import weakref
from typing import Dict, List, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage

import os
import sys
from collections import OrderedDict
from fvcore.nn.precise_bn import get_bn_modules
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.engine import hooks
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase

from dt2trainer import DefaultClutterizedTrainer
import datetime

import datetime

ct = datetime.datetime.now()
timestamp = str(ct).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")

print("timestamp: ", timestamp)


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

args = parser.parse_args()
        



    
    
        
        
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
cfg.INPUT.MASK_FORMAT = "bitmask"

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # only has one class (ballon)

cfg.OUTPUT_DIR = './output_'+timestamp

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("Declaring Trainer")
trainer = DefaultClutterizedTrainer(cfg)
print("Loading COCO weights")
trainer.resume_or_load(resume=False)
print("Starting Training") 
trainer.train()

