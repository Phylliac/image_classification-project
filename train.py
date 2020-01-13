  
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import project_functions as pf

ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('--data_dir',dest= 'data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)



pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
device = pa.gpu
epochs = pa.epochs

def main():
    train_datasets, valid_datasets, test_datasets = pf.data_transforms(root)
    trainloader, validloader, testloader = pf.data_loader(root)
    model, criterion, optimizer= pf.network_constructor(structure,dropout,lr,device)
    pf.model_trainer(model, criterion, optimizer, train_loader= trainloader, valid_loader = validloader)
    pf.checkpoint_saver(model,train_datasets = train_datasets )
    print('Voila! The model has been trained and saved!')
    
    
if __name__== "__main__":
    main()