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
import cv2 as cv



split_set = {'1':[2,3,4,5],
             '2':[1,3,4,5],
             '3':[1,2,4,5],
             '4':[1,2,3,5],
             '5':[1,2,3,4]
             }
split = torch.arange(500).view(5,100)

def take_tsk_photo(test_list):
    #print(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"))

    #print(test_list[0],test_list[1])
    image1 = test_list[0]
    image2 = test_list[1]
    image1 = transforms.ToTensor()(image1)
    image2 = transforms.ToTensor()(image2)
   

    
    image = torch.stack((image1, image2))

    # image_pair = [sub_floder + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg' + ' && ' + sub_floder + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    return image,test_list


class tskinface_test_dataset(Dataset):
    def __init__(self,test_list):
                self.test_list = test_list
                
               

    def __getitem__(self, idx):
                return take_tsk_photo(self.test_list[idx])
                # print(idx)
                # import time
                #time.sleep(0.5)
                #print(self.son+self.dau+self.son+self.dau)
                # if idx in range(0,self.son) : #(0,411)
                #     return take_tsk_photo(sub_floder='fs',root = self.root,train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                # if idx in range(self.son,self.son+ self.son) :
                #     return take_tsk_photo(sub_floder='ms', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                # if idx in range(self.son + self.son, self.son+self.son+self.dau):
                #     return take_tsk_photo(sub_floder='fd', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                # if idx in range(self.son+self.son+self.dau, self.son+self.son+self.dau+self.dau):
                #     return take_tsk_photo(sub_floder='md', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                # if idx >=  self.son+self.son+self.dau+self.dau:
                #     jdx = idx - (self.son+self.son+self.dau+self.dau)
                #     if jdx in range(0,self.son) :
                #             return take_tsk_photo(sub_floder='fs',root = self.root,train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                #     if jdx in range(self.son,self.son+ self.son) :
                #         return take_tsk_photo(sub_floder='ms', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                #     if jdx in range(self.son + self.son, self.son+self.son+self.dau):
                #         return take_tsk_photo(sub_floder='fd', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                #     if jdx in range(self.son+self.son+self.dau, self.length):
                #         return take_tsk_photo(sub_floder='md', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])


    def __len__(self):
        return len(self.test_list)



