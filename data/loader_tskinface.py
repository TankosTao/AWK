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


import yaml

configs = yaml.load(open('./configs/configs.yml', 'rb'), Loader=yaml.Loader)
picroot = configs['TSKINFACE']['picroot']


split_set = {'1':[2,3,4,5],
             '2':[1,3,4,5],
             '3':[1,2,4,5],
             '4':[1,2,3,5],
             '5':[1,2,3,4]
             }
split = torch.arange(500).view(5,100)


imgdic = {}
imgdic.update({picroot:{}})



for path_1 in os.listdir(picroot):
    print("#############",path_1)

    if  os.path.isdir(os.path.join(picroot,path_1)):
        
        if path_1 not in imgdic[picroot].keys():
            imgdic[picroot].update({path_1:{}})
        #print(imgdic)
        for path_2 in os.listdir(os.path.join(picroot,path_1)):
            if not path_2.endswith(".jpg"):
                continue
            if path_2 not in imgdic[picroot][path_1].keys():
                imgdic[picroot][path_1].update({path_2:cv.imread(os.path.join(picroot,path_1,path_2))})
             

def take_tsk_photo(sub_floder,root,train_idx,make_train_idx):
    #print(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"))

    if train_idx == make_train_idx:
        image_label = 1
        image1 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"))
        image2 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[1])+".jpg"))
        image_pair = [os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg") + ' && ' + os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[1])+".jpg")]


    else:
        image1 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"))
        image2 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(make_train_idx)+"_"+str(sub_floder[1])+".jpg"))

        image_label = 0
        image_pair = [os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg") + ' && ' + os.path.join(root,sub_floder,sub_floder+"_"+str(make_train_idx)+"_"+str(sub_floder[1])+".jpg")]

    
    image1 = transforms.ToTensor()(image1)
    image2 = transforms.ToTensor()(image2)
    image = torch.stack((image1, image2))

    # image_pair = [sub_floder + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg' + ' && ' + sub_floder + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    return image, image_label, image_pair

def take_tsk_photo_from_memory(sub_floder,root,train_idx,make_train_idx):
    #print(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"))

    if train_idx == make_train_idx:
        image_label = 1
        image1 = imgdic[root][sub_floder][sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"]
        image2 = imgdic[root][sub_floder][sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[1])+".jpg"]
        # image1 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"))
        # image2 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[1])+".jpg"))
        image_pair = [os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg") + ' && ' + os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[1])+".jpg")]


    else:
        image1 = imgdic[root][sub_floder][sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"]
        image2 = imgdic[root][sub_floder][sub_floder+"_"+str(make_train_idx)+"_"+str(sub_floder[1])+".jpg"]
        
        # image1 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg"))
        # image2 = cv.imread(os.path.join(root,sub_floder,sub_floder+"_"+str(make_train_idx)+"_"+str(sub_floder[1])+".jpg"))

        image_label = 0
        image_pair = [os.path.join(root,sub_floder,sub_floder+"_"+str(train_idx)+"_"+str(sub_floder[0])+".jpg") + ' && ' + os.path.join(root,sub_floder,sub_floder+"_"+str(make_train_idx)+"_"+str(sub_floder[1])+".jpg")]

    
    image1 = transforms.ToTensor()(image1)
    image2 = transforms.ToTensor()(image2)
    image = torch.stack((image1, image2))

    # image_pair = [sub_floder + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg' + ' && ' + sub_floder + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    return image, image_label, image_pair

class my_train_dataset(Dataset):
    def __init__(self,train_index,make_train_idx,son,dau,picroot):
                self.TRAIN_INDEX = train_index
                self.make_train_idx = make_train_idx
                self.length = len(self.TRAIN_INDEX)
                self.root = picroot
                self.son = son
                self.dau = dau
                # self.imgdic = imgdic

               

    def __getitem__(self, idx):
                # print(idx)
                # import time
                #time.sleep(0.5)
                #print(self.son+self.dau+self.son+self.dau)
                if idx in range(0,self.son) : #(0,411)
                    return take_tsk_photo_from_memory(sub_floder='fs',root = self.root,train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                if idx in range(self.son,self.son+ self.son) :
                    return take_tsk_photo_from_memory(sub_floder='ms', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                if idx in range(self.son + self.son, self.son+self.son+self.dau):
                    return take_tsk_photo_from_memory(sub_floder='fd', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])
                if idx in range(self.son+self.son+self.dau, self.son+self.son+self.dau+self.dau):
                    return take_tsk_photo_from_memory(sub_floder='md', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.make_train_idx[idx])

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
        return len(self.TRAIN_INDEX)

class tskinface_test_dataset(Dataset):
    def __init__(self,train_index,make_train_idx,son,dau,picroot,type):
                self.TRAIN_INDEX = train_index
                self.make_train_idx = make_train_idx
                self.length = len(self.TRAIN_INDEX)
                self.root = picroot
                self.son = son
                self.dau = dau
                self.type = type
                # imgdic = {}
                # imgdic.update({picroot:{}})
                # for path_1 in os.listdir(picroot):
        
                #     if  os.path.isdir(os.path.join(picroot,path_1)):
                        
                #         if path_1 not in imgdic[picroot].keys():
                #             imgdic[picroot].update({path_1:{}})
                #         #print(imgdic)
                #         for path_2 in os.listdir(os.path.join(picroot,path_1)):
                #             if not path_2.endswith(".jpg"):
                #                 continue
                #             if path_2 not in imgdic[picroot][path_1].keys():
                #                 imgdic[picroot][path_1].update({path_2:cv.imread(os.path.join(picroot,path_1,path_2))})
                # self.imgdic = imgdic
                
               

    def __getitem__(self, idx):
            

                return take_tsk_photo_from_memory(self.type,self.root,self.TRAIN_INDEX[idx],self.make_train_idx[idx])
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
        return len(self.TRAIN_INDEX)

def make_pair(collection):
    resultList = []
    seqList = collection.copy()
    # print(type(seqList))
    for i in collection:
        #print(i)
        index = (int)(np.random.rand(1)[0] * len(seqList))
        #print(index)
        while(seqList[index] ==i):
            #print(index)

        # index = 0 # for DEBUG ONLY
            index = (int)(np.random.rand(1)[0] * len(seqList))
        resultList.append(seqList[index])
        seqList.remove(seqList[index])
        #print(collection)
        #print()
    # print resultList
    return resultList

FMD_COLLECTION = {'split_1': list(range(1,101)),
                  'split_2': list(range(101,201)),
                  'split_3': list(range(201, 301)),
                  'split_4': list(range(301, 401)),
                  'split_5': list(range(401, 503))}
FMS_COLLECTION = {'split_1': list(range(1,103)),
                  'split_2': list(range(103,205)),
                  'split_3': list(range(205, 307)),
                  'split_4': list(range(307, 409)),
                  'split_5': list(range(409, 514))}


def get_train_loader(type,i,split_set,BATCH_SIZE,picroot):


    print(split_set[str(i)][0])
    print(split_set[str(i)][1])
    print(split_set[str(i)][2])
    print(split_set[str(i)][3])
    train_index_for_son =   FMS_COLLECTION['split_' + str(split_set[str(i)][0])] \
                        + FMS_COLLECTION['split_' + str(split_set[str(i)][1])] \
                        + FMS_COLLECTION['split_' + str(split_set[str(i)][2])] \
                        + FMS_COLLECTION['split_' + str(split_set[str(i)][3])]
    print('train_index_for_son:', len(train_index_for_son))
    train_index_for_dau =   FMD_COLLECTION['split_' + str(split_set[str(i)][0])] \
                        + FMD_COLLECTION['split_' + str(split_set[str(i)][1])]\
                        + FMD_COLLECTION['split_' + str(split_set[str(i)][2])] \
                        + FMD_COLLECTION['split_' + str(split_set[str(i)][3])]
    print('train_index_for_dau:', len(train_index_for_dau))
    MAKE_train_index_for_fs = make_pair(train_index_for_son)
    print('MAKE_train_index_for_fs:', len(MAKE_train_index_for_fs))
    MAKE_train_index_for_ms = make_pair(train_index_for_son)
    print('MAKE_train_index_for_ms:', len(MAKE_train_index_for_ms))
    MAKE_train_index_for_fd = make_pair(train_index_for_dau)
    print('MAKE_train_index_for_fd:', len(MAKE_train_index_for_fd))
    MAKE_train_index_for_md = make_pair(train_index_for_dau)
    print('MAKE_train_index_for_md:', len(MAKE_train_index_for_md))
    train_index = train_index_for_son + train_index_for_son + train_index_for_dau + train_index_for_dau

    test_index_for_son = FMS_COLLECTION['split_'+str(i)]
    print('test_index_for_son:', len(test_index_for_son))
    test_index_for_dau = FMD_COLLECTION['split_'+str(i)]
    print('test_index_for_dau:', len(test_index_for_dau))

    # MAKE_test_index_for_fs = list(reversed(test_index_for_son))
    # print('MAKE_test_index_for_fs:', len(MAKE_test_index_for_fs))
    # MAKE_test_index_for_ms = list(reversed(test_index_for_son))
    # print('MAKE_test_index_for_ms:', len(MAKE_test_index_for_ms))
    # MAKE_test_index_for_fd = list(reversed(test_index_for_dau))
    # print('MAKE_test_index_for_fd:', len(MAKE_test_index_for_fd))
    # MAKE_test_index_for_md = list(reversed(test_index_for_dau))
    # print('MAKE_test_index_for_md:', len(MAKE_test_index_for_md))
    

    MAKE_test_index_for_fs = make_pair(test_index_for_son)
    print('MAKE_test_index_for_fs:', len(MAKE_test_index_for_fs))
    MAKE_test_index_for_ms = make_pair(test_index_for_son)
    print('MAKE_test_index_for_ms:', len(MAKE_test_index_for_ms))
    MAKE_test_index_for_fd = make_pair(test_index_for_dau)
    print('MAKE_test_index_for_fd:', len(MAKE_test_index_for_fd))
    MAKE_test_index_for_md = make_pair(test_index_for_dau)
    print('MAKE_test_index_for_md:', len(MAKE_test_index_for_md))

    # MAKE_train_index = MAKE_train_index_for_fs + MAKE_train_index_for_ms + MAKE_train_index_for_fd + MAKE_train_index_for_md
    # MAKE_train_index = MAKE_train_index + train_index
    # train_index = train_index + train_index
    MAKE_train_index = train_index

    print('train_index:', len(train_index))
    print('MAKE_train_index:', len(MAKE_train_index))
    print(train_index)

    #from tskface import my_train_dataset
    train_dataset = my_train_dataset(train_index, MAKE_train_index,len(train_index_for_son),len(train_index_for_dau),picroot)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader,test_index_for_son,test_index_for_dau,MAKE_test_index_for_fs,MAKE_test_index_for_ms,MAKE_test_index_for_fd,MAKE_test_index_for_md


