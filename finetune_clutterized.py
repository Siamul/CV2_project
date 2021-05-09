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
from datetime import datetime
import pickle
from clutterized_dataloader import clutterized_datasetLoader

if not os.path.exists('./clutterized_weights/'):
    os.mkdir('./clutterized_weights/')

# Detect if we have a GPU available
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# current date and time
now = datetime.now()

timestamp = str(datetime.timestamp(now))
print("timestamp =", timestamp)
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, help="Data directory path", default="./data/")
parser.add_argument("--maxIter", type=int, help="Total number of iterations to train", default=30000)
parser.add_argument("--batchSize", type=int, help="Size of Batch", default=16)
parser.add_argument("--featureExtract", type=int, help="Flag for feature extracting. When 0, we train the whole model and when 1, we train the added fc layer.", default=1)
parser.add_argument("--inputSize", type=int, help="Resolution for the background image (and therefore the cluttered image", default=800)
parser.add_argument("--objectSize", type=int, help="Resolution for each object image", default = 512)
parser.add_argument("--optimType", type=str, help="Select between SGD and Adam", default="SGD")
parser.add_argument("--learningRate", type=float, help="Set the learning rate", default=0.001)
parser.add_argument("--momentum", type=float, help="Set the momentum for SGD optimizer", default=0.9)

args = parser.parse_args()

#Location of dataset
data_dir = args.dataDir

# Batch size for training (change depending on how much memory you have)
batch_size = args.batchSize

# Number of iterations to train for
max_iterations = args.maxIter

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = args.featureExtract

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_clutterized(model, clutterized_dataloader, criterion, optimizer, max_iterations, sigmoidThreshold=0.5):
    iterations = 0
    avg_loss_old = 99999
    running_loss = 0.0
    running_corrects = 0
    total = 0
    while(iterations < max_iterations):
        iterations += 1
        inputs, labels, objects_batch, mask_batch = clutterized_dataloader.getBatch(multi_process=True)
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_copy = labels.clone().detach()
        
        #relaxing labels
        labels *= 0.8
        labels[labels == 0] = 0.2 
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            preds = outputs.clone().detach()
            preds[preds>=sigmoidThreshold] = 1
            preds[preds<sigmoidThreshold] = 0
            
            loss.backward()
            optimizer.step()
        # statistics
        running_loss += loss.item()
        diffpredsout = torch.abs(preds - labels_copy)
        #diffpredsout[diffpredsout<0] = 1
        running_corrects += torch.numel(diffpredsout) - torch.sum(diffpredsout)
        total += torch.numel(diffpredsout)
        print('Loss in iteration {}/{}: {:.4f}'.format(iterations, max_iterations, loss.item()))
        if iterations%100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_corrects.double() / total
            print('##############################################################################################')
            print('Loss averaged over 100 iterations: {:.4f} Total accuracy over 100 iterations: {:.4f}'.format(avg_loss, avg_acc))
            print('##############################################################################################')
            if avg_loss_old > avg_loss:
                best_acc = avg_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                states = {
                    'state_dict': best_model_wts,
                    'optimizer': optimizer.state_dict(),
                    'id_to_class': clutterized_dataloader.get_idx_to_class()
                }
                torch.save(states, "./clutterized_weights/model_"+timestamp+"_"+str(best_acc.item())+".pkl")
                avg_loss_old = avg_loss
            running_loss = 0.0
            running_corrects = 0
            total = 0


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract != 0:
        print("Extracting features, not training entire network")
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        print("Training entire network")
        for name, param in model.named_parameters():
            param.requires_grad = True
    return model
#Define the model
model = models.resnet50(pretrained=True)
#print(model)

#Ensure that the model has no gradient if we want to finetune
model = set_parameter_requires_grad(model, feature_extract)

#Define Clutterized Dataloader (it loads batch of images with a randomly selected background and clutter of objects
input_size = args.inputSize
object_size = args.objectSize
clutterized_dataloader = clutterized_datasetLoader(dataset_location = './Dataset/', device_str=device_str, input_size=input_size, object_size=object_size, batch_size=batch_size)
num_classes = clutterized_dataloader.get_num_classes()

#Add the final layers to the model
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
              nn.Linear(num_ftrs, num_classes),
              nn.Sigmoid()
           )

#Move model to appropriate device
model = model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract != 0:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
learning_rate = args.learningRate
momentum = args.momentum
optim_type = args.optimType.lower().strip()
print('Using optimizer: ', optim_type)
if optim_type == 'sgd':
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
elif optim_type == 'adam':
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

# Setup the loss fxn
criterion = nn.BCELoss()

# Train and evaluate
model, hist = train_model_clutterized(model, clutterized_dataloader, criterion, optimizer, max_iterations)