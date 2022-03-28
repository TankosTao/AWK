import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# a ="""

# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# """


class CNNEncoder_14(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder_14, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        #out = out.view(out.size(0),-1)
        #print(out1.size())
        #print(out4.size())

        return out1,out4 # 64

class RelationNetwork_14(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork_14, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(256,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*6*6,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
#
        diffX = x1.size()[3] - x2.size()[3]
#
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
#
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.size())
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class CNNEncoder_1234(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder_1234, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        #out = out.view(out.size(0),-1)
        #print(out1.size())
        #print(out2.size())
        #print(out3.size())
        #print(out4.size())

        return out1,out2,out3,out4 # 64

class RelationNetwork_1234(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork_1234, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*6*6,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x1,x2,x3,x4):
        #diffY12 = x1.size()[2] - x2.size()[2]
#
        #diffX12 = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [8, 8, 8, 8])
        x3 = F.pad(x3, [8, 8, 8, 8])
        x4 = F.pad(x4, [9, 9, 9, 9])

        x = torch.cat([x1, x2,x3,x4], dim=1)
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.size())
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out