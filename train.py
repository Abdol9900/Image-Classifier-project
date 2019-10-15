
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
import torch.nn.functional as F
from os import listdir
import pandas as pd
import argparse
from collections import OrderedDict

parser= argparse.ArgumentParser()


parser.add_argument('data_dir',action='store', default='./flowers/',type=str)
parser.add_argument('--save_dir', action='store',type=str)
parser.add_argument('-lr','--learning_rate', action='store', default=0.001,type=float)
parser.add_argument('-H','--hidden_units', type=int, action='store', default=90)
parser.add_argument('-e','--epochs', action='store',default=13,type=int)
parser.add_argument('-ar','--arch', action='store', default='vgg13', type=str)
parser.add_argument('-gp','--gpu',action='store_true',help='Use GPU if available')

arg = parser.parse_args()
arch = arg.arch
learning_rate = arg.learning_rate 
epochs = arg.epochs
hidden_units = arg.hidden_units
save_dir = arg.save_dir
data_dir = arg.data_dir

if arg.gpu:        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
test_data= datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
# TODO: Build and train your network
 
model = getattr(models,arch)(pretrained=True)
in_features = model.classifier[0].in_features

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
    classifier= nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, arg.hidden_units)),                       
                              ('fc2', nn.Linear(arg.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))

    
        
    model.classifier = classifier

#  defining optimizer and criterion
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


running_loss=0
steps = 0
model.cuda()
for i in range(epochs):
    
    for ii, (inputs,labels) in enumerate(trainloader):
        steps+=1
         # move images and labels to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  
        
    else:
       
        valid_loss = 0
        accuracy = 0
        model.eval()     
        with torch.no_grad():
             
            for ii, (inputs_2, labels_2) in enumerate(validloader):
                steps+=1
                inputs_2, labels_2 = inputs_2.to(device), labels_2.to(device)
                log_ps = model.forward(inputs_2)
                valid_loss+= criterion(log_ps, labels_2)
                    
                    
                # Calculate accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels_2.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        model.train() 
  
        print('Epoch: {}/{}.. '.format(i+1, epochs),
              'Traning Loss: {:.3f}.. '.format(running_loss/len(trainloader)),
              'validation Loss: {:.3f}.. '.format(valid_loss/len(validloader)),
                'Accuracy: {:.3f}'.format(accuracy/len(testloader)))
        running_loss = 0      

model.to ('cpu') 

model.class_to_idx = train_data.class_to_idx 

checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'arch': arch,
              'class_to_idx':model.class_to_idx,
              'number_epochs': epochs,
			  'hidden_units': hidden_units
             }
			 
if arg.save_dir:
    save_dir = arg.save_dir
else:   
    save_dir = 'checkpoint.pth'
			 
torch.save(checkpoint, save_dir)    