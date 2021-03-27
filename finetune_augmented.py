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
import random
import pickle
from datetime import datetime

# current date and time
now = datetime.now()

timestamp = str(datetime.timestamp(now))
print("timestamp =", timestamp)

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, help="Data directory path", default="./data/")
parser.add_argument("--epochs", type=int, help="Number of epoch to train", default=50)
parser.add_argument("--batchSize", type=int, help="Size of Batch", default=8)
parser.add_argument("--numClasses", type=int, help="Number of classes in the dataset", default=11)
parser.add_argument("--featureExtract", type=bool, help="Flag for feature extracting. When False, we finetune the whole model", default=True)

args = parser.parse_args()

data_dir = args.dataDir

# Number of classes in the dataset
num_classes = args.numClasses

# Batch size for training (change depending on how much memory you have)
batch_size = args.batchSize

# Number of epochs to train for
num_epochs = args.epochs

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = args.featureExtract


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, input_size=512, sigmoidThreshold=0.5, num_classes=11):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total = 0

            # Iterate over data.
            
            #for inputs, labels in dataloaders[phase]:
                #inputs = inputs.to(device)
                #labels = labels.to(device)
            data_iter = iter(dataloaders[phase])
            i = 0
            while(i < len(dataloaders[phase])):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    concat_factor = random.choice([1,4,9])
                    if concat_factor == 1:
                        inputs, labels = next(data_iter)
                        labels = nn.functional.one_hot(labels, num_classes)
                        i += 1
                    elif concat_factor == 4:
                        try:
                            data1, labels1 = next(data_iter)
                            labels1_one_hot = nn.functional.one_hot(labels1, num_classes)
                            try:
                                data2, labels2 = next(data_iter)
                                labels2_one_hot = nn.functional.one_hot(labels2, num_classes)
                                assert labels2_one_hot.shape == labels1_one_hot.shape
                                try:
                                    data3, labels3 = next(data_iter)
                                    labels3_one_hot = nn.functional.one_hot(labels3, num_classes)
                                    assert labels3_one_hot.shape == labels1_one_hot.shape
                                    try:
                                        data4, labels4 = next(data_iter)
                                        labels4_one_hot = nn.functional.one_hot(labels4, num_classes)
                                        assert labels4_one_hot.shape == labels1_one_hot.shape
                                        data_part1 = torch.cat((data1, data2), 2)
                                        data_part2 = torch.cat((data3, data4), 2)
                                        inputs = torch.cat((data_part1, data_part2), 3)
                                    except Exception as e:
                                        inputs = torch.cat((data1, data2, data3), 2)
                                        labels4_one_hot = torch.zeros(labels1_one_hot.shape)
                                except Exception as e:
                                    inputs = torch.cat((data1, data2), 2)
                                    labels3_one_hot = torch.zeros(labels1_one_hot.shape)
                                    labels4_one_hot = torch.zeros(labels1_one_hot.shape)
                            except Exception as e:
                                inputs = data1
                                labels2_one_hot = torch.zeros(labels1_one_hot.shape)
                                labels3_one_hot = torch.zeros(labels1_one_hot.shape)
                                labels4_one_hot = torch.zeros(labels1_one_hot.shape)
                        except Exception as e:
                            break
                        
                        i += 4
                        
                        labels = labels1_one_hot + labels2_one_hot + labels3_one_hot + labels4_one_hot
                        labels[labels>1] = 1
                        
                    elif concat_factor == 9:
                        try:
                            data1, labels1 = next(data_iter)
                            labels1_one_hot = nn.functional.one_hot(labels1, num_classes)
                            try:
                                data2, labels2 = next(data_iter)
                                labels2_one_hot = nn.functional.one_hot(labels2, num_classes)
                                assert labels2_one_hot.shape == labels1_one_hot.shape
                                try:
                                    data3, labels3 = next(data_iter)
                                    labels3_one_hot = nn.functional.one_hot(labels3, num_classes)
                                    assert labels3_one_hot.shape == labels1_one_hot.shape
                                    data_part1 = torch.cat((data1, data2, data3), 2)
                                    try: 
                                        data4, labels4 = next(data_iter)
                                        labels4_one_hot = nn.functional.one_hot(labels4, num_classes)
                                        assert labels4_one_hot.shape == labels1_one_hot.shape
                                        try:
                                            data5, labels5 = next(data_iter)
                                            labels5_one_hot = nn.functional.one_hot(labels5, num_classes)
                                            assert labels5_one_hot.shape == labels1_one_hot.shape
                                            try: 
                                                data6, labels6 = next(data_iter)
                                                labels6_one_hot = nn.functional.one_hot(labels6, num_classes)
                                                assert labels6_one_hot.shape == labels1_one_hot.shape
                                                data_part2 = torch.cat((data1, data2, data3), 2)
                                                try: 
                                                    data7, labels7 = next(data_iter)
                                                    labels7_one_hot = nn.functional.one_hot(labels7, num_classes)
                                                    assert labels7_one_hot.shape == labels1_one_hot.shape
                                                    try:
                                                        data8, labels8 = next(data_iter)
                                                        labels8_one_hot = nn.functional.one_hot(labels8, num_classes)
                                                        assert labels8_one_hot.shape == labels1_one_hot.shape
                                                        try: 
                                                            data9, labels9 = next(data_iter)
                                                            labels9_one_hot = nn.functional.one_hot(labels9, num_classes)
                                                            assert labels9_one_hot.shape == labels1_one_hot.shape
                                                            data_part3 = torch.cat((data7, data8, data9), 2)
                                                            inputs = torch.cat((data_part1, data_part2, data_part3), 3)
                                                        except Exception as e:
                                                            inputs = torch.cat((data_part1, data_part2), 3)
                                                            labels7_one_hot = torch.zeros(labels1_one_hot.shape)
                                                            labels8_one_hot = torch.zeros(labels1_one_hot.shape)
                                                            labels9_one_hot = torch.zeros(labels1_one_hot.shape)
                                                    except Exception as e:
                                                        inputs = torch.cat((data_part1, data_part2), 3)
                                                        labels7_one_hot = torch.zeros(labels1_one_hot.shape)
                                                        labels8_one_hot = torch.zeros(labels1_one_hot.shape)
                                                        labels9_one_hot = torch.zeros(labels1_one_hot.shape)
                                                except Exception as e:
                                                    inputs = torch.cat((data_part1, data_part2), 3)
                                                    labels7_one_hot = torch.zeros(labels1_one_hot.shape)
                                                    labels8_one_hot = torch.zeros(labels1_one_hot.shape)
                                                    labels9_one_hot = torch.zeros(labels1_one_hot.shape)
                                            except Exception as e:
                                                inputs = data_part1
                                                labels4_one_hot = torch.zeros(labels1_one_hot.shape)
                                                labels5_one_hot = torch.zeros(labels1_one_hot.shape)
                                                labels6_one_hot = torch.zeros(labels1_one_hot.shape)
                                                labels7_one_hot = torch.zeros(labels1_one_hot.shape)
                                                labels8_one_hot = torch.zeros(labels1_one_hot.shape)
                                                labels9_one_hot = torch.zeros(labels1_one_hot.shape)
                                        except Exception as e:
                                            inputs = data_part1
                                            labels4_one_hot = torch.zeros(labels1_one_hot.shape)
                                            labels5_one_hot = torch.zeros(labels1_one_hot.shape)
                                            labels6_one_hot = torch.zeros(labels1_one_hot.shape)
                                            labels7_one_hot = torch.zeros(labels1_one_hot.shape)
                                            labels8_one_hot = torch.zeros(labels1_one_hot.shape)
                                            labels9_one_hot = torch.zeros(labels1_one_hot.shape)
                                    except Exception as e:
                                        inputs = data_part1
                                        labels4_one_hot = torch.zeros(labels1_one_hot.shape)
                                        labels5_one_hot = torch.zeros(labels1_one_hot.shape)
                                        labels6_one_hot = torch.zeros(labels1_one_hot.shape)
                                        labels7_one_hot = torch.zeros(labels1_one_hot.shape)
                                        labels8_one_hot = torch.zeros(labels1_one_hot.shape)
                                        labels9_one_hot = torch.zeros(labels1_one_hot.shape)
                                except Exception as e:
                                    break
                            except Exception as e:
                                break
                        except Exception as e:
                            break
                                                
                        labels = labels1_one_hot + labels2_one_hot + labels3_one_hot + labels4_one_hot + \
                                 labels5_one_hot + labels6_one_hot + labels7_one_hot + labels8_one_hot + labels9_one_hot
                        labels[labels>1] = 1
                        i += 9
                    inputs = nn.functional.interpolate(inputs, size=input_size)
                    inputs = inputs.to(device)
                    labels = labels.to(torch.float32)
                    labels = labels.to(device)
                    print(phase+" iteration "+str(i)+" using "+str(concat_factor)+" image.")
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    preds = outputs.clone().detach()
                    preds[preds>=sigmoidThreshold] = 1
                    preds[preds<sigmoidThreshold] = 0

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                diffpredsout = preds - labels
                diffpredsout[diffpredsout<0] = 1
                running_corrects += torch.numel(diffpredsout) - torch.sum(diffpredsout)
                total += torch.numel(diffpredsout)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / total

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "./augmented_weights/model_"+timestamp+"_"+str(best_acc.item())+".pt")
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model = models.resnet50(pretrained=True)
print(model)

import torch.nn as nn
set_parameter_requires_grad(model, feature_extract)
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

data_transforms = transforms.Compose([        
        transforms.RandomRotation(180),
        transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()
])

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['train', 'val']}
image_dataset = datasets.ImageFolder(data_dir, data_transforms)
with open("augmented_class_to_idx_"+timestamp+".pkl", "wb") as outfile:
    pickle.dump(image_dataset.class_to_idx , outfile)
train_size = int(0.85 * len(image_dataset))
validation_size = len(image_dataset) - train_size
image_datasets_index = torch.utils.data.dataset.random_split(image_dataset, [train_size, validation_size])
image_datasets = {}
image_datasets['train'] = image_datasets_index[0]
image_datasets['val'] = image_datasets_index[1]

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
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
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.BCELoss()

# Train and evaluate
model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=False, input_size = input_size)
