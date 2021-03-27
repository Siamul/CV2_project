import argparse
import cv2
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from GradCAMComponents import GradCam, GuidedBackpropReLUModel, preprocess_image, show_cam_on_image, deprocess_image

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, help="image for gradcam", default="image.jpg")
parser.add_argument("--numClasses", type=int, help="Number of classes for the model", default=11)
parser.add_argument("--modelWeights", type=str, help="Location of model weights", default="./model_weights.pt")
parser.add_argument("--modelType", type=str, help="Type of model for GradCAM [normal, regularized, augmented]", default="augmented")

args = parser.parse_args()

model = models.resnet50()
num_classes = args.numClasses

if args.modelType == "normal":
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    print("Model Type: normal")
elif args.modelType == "regularized":
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
             nn.Dropout(0.5),
             nn.Linear(num_ftrs, num_classes * 2),
           )
    input_size = 224
    print("Model Type: regularized")
elif args.modelType == "augmented":
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
             nn.Dropout(0.2),
             nn.Linear(num_ftrs, num_classes * 2),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(num_classes * 2, num_classes),
             nn.Sigmoid()
           )
    input_size = 512
    print("Model Type: augmented")
    
model.load_state_dict(torch.load(args.modelWeights))
#for param in model.parameters():
#    param.requires_grad = True


grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=True, use_softmax=True)

img = cv2.imread(args.image, 1)
#img = Image.open(BytesIO(uploaded['Image_file_name.jpg']))
img = np.float32(img) / 255

orig_width = img.shape[1]
orig_height = img.shape[0]

new_width = 256
new_height = int(((orig_height * 256) / orig_width))

dsize = (new_width, new_height)

img = cv2.resize(img, dsize)

# Opencv loads as BGR:
img = img[:, :, ::-1]
input_img = preprocess_image(img)

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
for target_category in range(num_classes):
    model = models.resnet50()
    num_classes = args.numClasses

    if args.modelType == "normal":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        print("Model Type: normal")
    elif args.modelType == "regularized":
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
             nn.Dropout(0.5),
             nn.Linear(num_ftrs, num_classes * 2),
           )
        input_size = 224
        print("Model Type: regularized")
    elif args.modelType == "augmented":
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
             nn.Dropout(0.2),
             nn.Linear(num_ftrs, num_classes * 2),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(num_classes * 2, num_classes),
             nn.Sigmoid()
           )
        input_size = 512
        print("Model Type: augmented")
    
    model.load_state_dict(torch.load(args.modelWeights))
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=True, use_softmax=True)
    input_img = preprocess_image(img)
    grayscale_cam, output_vec = grad_cam(input_img, target_category)
    
    print(output_vec)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)
    
    if not os.path.exists("./outputs"):
        os.mkdir("./outputs/")
    if not os.path.exists("./outputs/"+ str(target_category) +"/"):
        os.mkdir("./outputs/"+ str(target_category)+"/")
    cv2.imwrite("./outputs/" + str(target_category) +"/cam.jpg", cam)
    cv2.imwrite("./outputs/" + str(target_category) +"/gb.jpg", gb)
    cv2.imwrite("./outputs/" + str(target_category) +"/cam_gb.jpg", cam_gb)
