import torch
import os.path
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import input_args_train as args

#directories
data_dir = args.result.training_data
train_dir =  os.path.join(data_dir,'train')
valid_dir = os.path.join(data_dir,'valid')

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.RandomRotation(10),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_datasets,batch_size=64,shuffle=True)