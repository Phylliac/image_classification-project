# This fiile outlines the various functions to be used in both predict.py and train.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

#The transform function tahes the root directory as argument and returns the train, test and validation datasets

def data_transforms(root_directory):
   data_dir = root_directory
   train_dir = data_dir + '/train'
   valid_dir = data_dir + '/valid'
   test_dir = data_dir + '/test' 
    
     #Define your transforms for the training, validation, and testing sets
    
   train_transforms = transforms.Compose([
                                            transforms.Resize(255),
                                            transforms.CenterCrop(244),
                                            transforms.RandomRotation(20),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])
   valid_transforms = transforms.Compose([
                                            transforms.Resize(255),
                                            transforms.CenterCrop(244),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])
   test_transforms = valid_transforms
    # TODO: Load the datasets with ImageFolder
    #DONE
   train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
   valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
   test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    
   return train_datasets, valid_datasets, test_datasets


def data_loader(root_directory):
    train_datasets, valid_datasets, test_datasets = data_transforms(root_directory)

    # Using the image datasets and the tranforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 32)
    
    return train_loader, valid_loader, test_loader

def network_constructor(structure='vgg16',dropout=0.5,lr = 0.001,device='gpu') :
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Please try for vgg16, alexnet or densenet121 only")

    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (arch[structure], 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (dropout)),
                            ('fc2', nn.Linear (4096, 2048)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (dropout)),
                            ('fc3', nn.Linear (2048, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam (model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
            model.cuda()

    return model, criterion, optimizer

def validation(model, valid_loader, criterion, device='gpu'):
    model.to ('cuda')
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        if torch.cuda.is_available() and device =='gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def model_trainer(model, criterion, optimizer, epochs = 7, print_every=40, train_loader=0, device='gpu', valid_loader=0):
    steps = 0
    for e in range (epochs): 
        running_loss = 0
        for ii, (inputs, labels) in enumerate (train_loader):
            steps += 1

            if torch.cuda.is_available() and device =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad () #where optimizer is working on classifier paramters only

            # Forward and backward passes
            outputs = model.forward (inputs) #calculating output
            loss = criterion (outputs, labels) #calculating loss
            loss.backward () 
            optimizer.step () #performs single optimization step 

            running_loss += loss.item () # loss.item () returns scalar value of Loss function

            if steps % print_every == 0:
                model.eval () #switching to evaluation mode so that dropout is turned off

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                      "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))

                running_loss = 0
                model.train()
                
def checkpoint_saver(model=0,path='checkpoint.pth',structure ='vgg16',dropout=0.5,lr=0.001,epochs=7, train_datasets=0):
    model.class_to_idx = train_datasets.class_to_idx
    model.cpu
    checkpoint = {
                  'structure': structure,
                  'epochs': epochs,
                  'lr': lr,
                  'dropout': dropout,
                  'state_dict' : model.state_dict(),
                  'mapping' : model.class_to_idx
                 }
    torch.save(checkpoint, path)
    
def load_checkpoint(path= 'checkpoint.pth'):
    trained_model = torch.load(path)
    model,_,_ = network_constructor(trained_model['structure'],  trained_model['dropout'], trained_model['lr'])
    model.class_to_idx = trained_model['mapping']
    model.load_state_dict(trained_model['state_dict'])
    
    return model    

def process_image(image ='/home/workspace/aipnd-project/flowers/test/1/image_06754.jpg'):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a pythorch tensor
    '''
    im = Image.open(image)
    im_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])
    new_im = im_transform(im)
    return new_im

def predict(image='/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg', model=0, topk=5,device='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and device =='gpu':
        model.to('cuda')
    img = process_image(image)
    img = img.unsqueeze_(0)
    img = img.float()
    
    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img)
            
    probability = F.softmax(output.data, dim =1)
    return probability.topk(topk)