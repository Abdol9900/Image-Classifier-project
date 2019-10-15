import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
import torch.nn.functional as F
import pandas as pd
import argparse
from PIL import Image
from collections import OrderedDict
from os import listdir
import json


parser = argparse.ArgumentParser()
parser.add_argument('data_dir',action='store', default='./flowers/',type=str)
parser.add_argument('checkpoint', action='store',type=str)
parser.add_argument('-img','--image_path',action='store',type=str,default='./flowers/test/4/image_05636.jpg')
parser.add_argument('-top', '--topk', action='store',type=int)
parser.add_argument('-g','--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('-ar','--arch', action='store', default='vgg19', type=str)

arg = parser.parse_args()

checkpoint = arg.checkpoint
image_path = arg.image_path
data_dir = arg.data_dir
topk = arg.topk
arch = arg.arch
if arg.gpu:
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
data_dir = arg.data_dir	
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)   

    
model = getattr(models,arch)(pretrained=True)
in_features = model.classifier[0].in_features

def loadcheckpoint(path='checkpoint.pth'):
    
    checkpoint =torch.load(path)
    model.state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    return model

loadcheckpoint('checkpoint.pth')
    

def process_image(image_path):
    im = Image.open(image_path)
    im_pre=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image_tensor= im_pre(im)
    
    return image_tensor 
   

def predict(image_path, model, topk=3):
  
    p_image= process_image(image_path)
    p_tensor = torch.tensor(p_image)
    p_tensor = p_tensor.float()
    p_tensor = p_tensor.unsqueeze(0)
    
   
    model.eval()
    Log_predictions = model.forward(p_tensor)
    predictions = torch.exp(Log_predictions)
    
   
    top__preds, top__labs = predictions.topk(3)
    
    top__preds = top__preds.detach().numpy().tolist()
    
    top__labs = top__labs.tolist()
    
    labels = pd.DataFrame({'classes':pd.Series(model.class_to_idx),'Flowers':pd.Series(cat_to_name)})
    labels = labels.set_index('classes')
    
    labels = labels.iloc[top__labs[0]]
    labels['predictions'] = top__preds[0]
    
    return labels

prediction= predict(image_path,model)
print(prediction)