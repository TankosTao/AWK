
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision.transforms.functional import rotate
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2 as cv
import yaml



def take_photo(train_img,label):
    image1 = cv.imread(train_img[0])
    image2 = cv.imread(train_img[1])
    image1 = transforms.ToTensor()(image1)
    image2 = transforms.ToTensor()(image2)

    image = torch.stack((image1, image2))
   
    image_label = label

    image_pair = train_img

    return image, image_label, image_pair


class train_dataset(Dataset):
            def __init__(self,train_file):
                # print(train_file)
                f = open(train_file,"r",encoding='utf-8')
                train_img_list = []
                labels = []
                for line in f.readlines():
                    img1,img2,label = line.strip().split("\t")
                    train_img_list.append([img1,img2])
                    labels.append(label)
                

                self.train_img_list = train_img_list
                self.labels = labels


            def __getitem__(self, idx):
                return take_photo(self.train_img_list[idx],self.labels[idx])
       

            def __len__(self):
                return len(self.train_img_list)

def get_train_loader_txt(train_file,BATCH_SIZE):
    
    
    TRAIN_DATASET = train_dataset(train_file)
    train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader
