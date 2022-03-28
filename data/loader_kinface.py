
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

configs = yaml.load(open('./configs/configs.yml', 'rb'), Loader=yaml.Loader)
picroot1 = configs['KINFACE1']['picroot']
picroot2 = configs['KINFACE2']['picroot']

imgdic = {}
imgdic.update({picroot1:{}})
imgdic.update({picroot2:{}})

for picroot in [picroot1,picroot2]:
    for path_2 in os.listdir(picroot):
        #print("#############",path_1)

        # if  os.path.isdir(os.path.join(picroot,path_1)):
            
        #     if path_1 not in imgdic[picroot].keys():
        #         imgdic[picroot].update({path_1:{}})
        #     #print(imgdic)
            # for path_2 in os.listdir(os.path.join(picroot,path_1)):
        if not path_2.endswith(".jpg"):
            continue
        if path_2 not in imgdic[picroot].keys():
            imgdic[picroot].update({path_2:cv.imread(os.path.join(picroot,path_2))})
    #print(imgdic[picroot].keys())

                

split_set = {'1':[2,3,4,5],
             '2':[1,3,4,5],
             '3':[1,2,4,5],
             '4':[1,2,3,5],
             '5':[1,2,3,4]
             }
split = torch.arange(500).view(5,100)



def take_photo(sub_floder,root,train_idx,make_train_idx):
    image1 = cv.imread(root + '/' + sub_floder  + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg')
    image2 = cv.imread(root + '/' + sub_floder  + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg')
    image1 = transforms.ToTensor()(image1)
    image2 = transforms.ToTensor()(image2)

    image = torch.stack((image1, image2))
    if train_idx == make_train_idx:
        image_label = 1

    else:
        image_label = 0

    image_pair = [sub_floder + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg' + ' && ' + sub_floder + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    return image, image_label, image_pair


def take_photo_from_memory_kinface1(sub_floder,root,train_idx,make_train_idx):
    # image1 = cv.imread(root + '/' + sub_floder  + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg')
    # image2 = cv.imread(root + '/' + sub_floder  + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg')
    
    image1 = imgdic[root][ sub_floder  + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg']
    image2 = imgdic[root][ sub_floder  + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    image1 = transforms.ToTensor()(image1)
    image2 = transforms.ToTensor()(image2)

    image = torch.stack((image1, image2))
    if train_idx == make_train_idx:
        image_label = 1

    else:
        image_label = 0

    image_pair = [sub_floder + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg' + ' && ' + sub_floder + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    return image, image_label, image_pair

def take_photo_from_memory_kinface2(sub_floder,root,train_idx,make_train_idx):
    # image1 = cv.imread(root + '/' + sub_floder  + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg')
    # image2 = cv.imread(root + '/' + sub_floder  + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg')
    image1 = imgdic[root][ sub_floder  + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg']
    image2 = imgdic[root][ sub_floder  + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    image1 = transforms.ToTensor()(image1)
    image2 = transforms.ToTensor()(image2)

    image = torch.stack((image1, image2))
    if train_idx == make_train_idx:
        image_label = 1

    else:
        image_label = 0

    image_pair = [sub_floder + '_' + str(train_idx).zfill(3) + '_' + '1' + '.jpg' + ' && ' + sub_floder + '_' + str(make_train_idx).zfill(3) + '_' + '2' + '.jpg']

    return image, image_label, image_pair

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


class My_dataset(Dataset):
    def __init__(self,MAKE_train_index,train_index,root):
        self.MAKE_TRAIN_INDEX = MAKE_train_index
        self.TRAIN_INDEX = train_index
        #print(len(train_index))
        #print(len(MAKE_train_index))
        self.root = root

    def __getitem__(self, idx):
        for sub_floder in ['fs','ms','fd','md','fs','ms','fd','md']:
            image1 = Image.open(self.root + '/' + sub_floder + '/' + sub_floder+'_'+str(self.TRAIN_INDEX[idx])+'_'+sub_floder[0]+'.jpg')
            image2 = Image.open(self.root + '/' + sub_floder + '/' + sub_floder+'_'+str(self.MAKE_TRAIN_INDEX[idx])+'_'+sub_floder[1]+'.jpg')
            image1 = image1.convert("RGB")
            image2 = image2.convert('RGB')
            #image1 = image1.convert('L')
            #image2 = image2.convert('L')
            image1 = transforms.ToTensor()(image1)
            image2 = transforms.ToTensor()(image2)
            image = torch.stack((image1, image2))
            if self.TRAIN_INDEX[idx] ==self.MAKE_TRAIN_INDEX[idx]:
                image_label = 1
            else:
                image_label = 0

            image_pair = [sub_floder+'_'+str(self.TRAIN_INDEX[idx])+'_'+sub_floder[0]+'.jpg' + ' && ' + sub_floder+'_'+str(self.MAKE_TRAIN_INDEX[idx])+'_'+sub_floder[1]+'.jpg']
            # print(sub_floder+'_'+str(self.TRAIN_INDEX[idx])+'_'+sub_floder[0]+'.jpg' + ' && ' + sub_floder+'_'+str(self.MAKE_TRAIN_INDEX[idx])+'_'+sub_floder[1]+'.jpg'+" "+str(image_label))
            return image, image_label,image_pair



    def __len__(self):
        return len(self.MAKE_TRAIN_INDEX)

def get_collettion(type):
    FMD_COLLECTION = {}
    FMS_COLLECTION = {}
    if type=='TSKINFACE':
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
                  
    
    return FMD_COLLECTION,FMS_COLLECTION

def load_kinface(type,dataroot):
    # DATA_SET FOR FMD
    COLLECTION = {}
    image_embedding_F_D = 'fd_pairs'

    # DATA_SET FOR FMS
    image_embedding_F_S = 'fs_pairs'

    # DATA_SET FOR FMSD
    image_embedding_M_D = 'md_pairs'

    # DATA_SET FOR FMSD
    image_embedding_M_S = 'ms_pairs'

    # LAOD Father
    matcontent_F_D = sio.loadmat(dataroot + "/" + image_embedding_F_D + ".mat")
    matcontent_F_S = sio.loadmat(dataroot + "/" + image_embedding_F_S + ".mat")

    # LOAD Mother
    matcontent_M_D = sio.loadmat(dataroot + "/" + image_embedding_M_D + ".mat")
    matcontent_M_S = sio.loadmat(dataroot + "/" + image_embedding_M_S + ".mat")

    feature_F_D = matcontent_F_D['pairs']
    print('F_D feature :', feature_F_D.shape)
    feature_F_S = matcontent_F_S['pairs']
    print('FDM_M feature :', feature_F_S.shape)
    feature_M_D = matcontent_M_D['pairs']
    print('M_D feature :', feature_M_D.shape)
    feature_M_S = matcontent_M_S['pairs']
    print("M_S feature :", feature_M_S.shape)
   
    if type =='KINFACE2':
    
        for i in range(1, 6):
            res_fs = []
            for j in feature_F_S:
                if j[0][0][0] == i:
                    if int(j[2][0][3:6]) not in res_fs:
                        res_fs.append(int(j[2][0][3:6]))
            COLLECTION['split_' + str(i)] = res_fs
        #(COLLECTION)
        return feature_F_D,feature_F_S,feature_M_D,feature_M_S,COLLECTION
    elif type=='KINFACE1':
        F_D_COLLECTION = {'split_1': list(range(1,28)),
                  'split_2': list(range(28,55)),
                  'split_3': list(range(55, 82)),
                  'split_4': list(range(82,109 )),
                  'split_5': list(range(109, 135))}

        F_S_COLLECTION = {'split_1': list(range(1,32)),
                        'split_2': list(range(32,65)),
                        'split_3': list(range(65, 97)),
                        'split_4': list(range(97,125 )),
                        'split_5': list(range(125, 157))}

        M_D_COLLECTION = {'split_1': list(range(1,26)),
                        'split_2': list(range(26,51)),
                        'split_3': list(range(51, 76)),
                        'split_4': list(range(76,102 )),
                        'split_5': list(range(102, 127))}
        M_S_COLLECTION = {'split_1': list(range(1,24)),
                        'split_2': list(range(24,47)),
                        'split_3': list(range(47, 70)),
                        'split_4': list(range(70,93 )),
                        'split_5': list(range(93, 111))}



        for i in range(1, 6):
                res_fs = []
                res_ms = []
                res_fd = []
                res_md = []
                for j in feature_F_S:
                    if j[0][0][0] == i:
                        if int(j[2][0][3:6]) not in res_fs:
                            res_fs.append(int(j[2][0][3:6]))
                            # print(res_fs)
                            # import time
                            # time.sleep(5)
                F_S_COLLECTION['split_' + str(i)] = res_fs

                for j in feature_M_S:
                    if j[0][0][0] == i:
                        if int(j[2][0][3:6]) not in res_ms:
                            res_ms.append(int(j[2][0][3:6]))
                M_S_COLLECTION['split_' + str(i)] = res_ms

                for j in feature_F_D:
                    if j[0][0][0] == i:
                        if int(j[2][0][3:6]) not in res_fd:
                            res_fd.append(int(j[2][0][3:6]))
                F_D_COLLECTION['split_' + str(i)] = res_fd

                for j in feature_M_D:
                    if j[0][0][0] == i:
                        if int(j[2][0][3:6]) not in res_md:
                            res_md.append(int(j[2][0][3:6]))
                M_D_COLLECTION['split_' + str(i)] = res_md
        # print(len(feature_F_D))
        # print(feature_F_D)
        # print(len(F_D_COLLECTION),F_D_COLLECTION)
       
    
        return feature_F_D,feature_F_S,feature_M_D,feature_M_S,F_S_COLLECTION,M_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION
class kinface2_train_dataset(Dataset):
            def __init__(self,train_index,root):
                self.TRAIN_INDEX = train_index
                self.length = len(self.TRAIN_INDEX)
                self.root = root
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
                if idx in range(0,self.length//4) :
                    return take_photo_from_memory_kinface2(sub_floder='fs',root = self.root,train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
                if idx in range(self.length//4,self.length//2) :
                    return take_photo_from_memory_kinface2(sub_floder='ms', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
                if idx in range(self.length//2,3*self.length//4):
                    return take_photo_from_memory_kinface2(sub_floder='fd', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
                if idx in range(3*self.length//4,self.length):
                    return take_photo_from_memory_kinface2(sub_floder='md', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
                # if idx in range(0,self.length//4) :
                #     return take_photo(sub_floder='fs',root = self.root,train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
                # if idx in range(self.length//4,self.length//2) :
                #     return take_photo(sub_floder='ms', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
                # if idx in range(self.length//2,3*self.length//4):
                #     return take_photo(sub_floder='fd', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
                # if idx in range(3*self.length//4,self.length):
                #     return take_photo(sub_floder='md', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])


            def __len__(self):
                return len(self.TRAIN_INDEX)


class kinface1_train_dataset(Dataset):
    def __init__(self,train_index,fs,ms,fd,md,root):
        self.TRAIN_INDEX = train_index
        self.length = len(self.TRAIN_INDEX)
        self.fs = fs
        self.ms = ms
        self.fd = fd
        self.md = md
        self.root = root
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
        if idx in range(0,self.fs) :
            return take_photo_from_memory_kinface1(sub_floder='fs',root = self.root,train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
        if idx in range(self.fs,self.fs+ self.ms) :
            return take_photo_from_memory_kinface1(sub_floder='ms', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
        if idx in range(self.fs + self.ms, self.fs+self.ms+self.fd):
            return take_photo_from_memory_kinface1(sub_floder='fd', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])
        if idx in range(self.fs+self.ms+self.fd, self.length):
            return take_photo_from_memory_kinface1(sub_floder='md', root=self.root, train_idx=self.TRAIN_INDEX[idx], make_train_idx=self.TRAIN_INDEX[idx])

    def __len__(self):
        return len(self.TRAIN_INDEX)
class kinface_test_dataset(Dataset):
        def __init__(self,feature_pairs,root):
            self.feature_pairs = feature_pairs
            self.root = root
            
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
            
            # print(self.root + '/' + str(self.feature_pairs[idx][2])[2:-2])

            #image1 = cv.imread(self.root + '/' + str(self.feature_pairs[idx][2])[2:-2])
            #image2 = cv.imread(self.root + '/' + str(self.feature_pairs[idx][3])[2:-2])
            image1 = imgdic[self.root][ str(self.feature_pairs[idx][2])[2:-2]]
            image2 = imgdic[self.root][ str(self.feature_pairs[idx][3])[2:-2]]
            #print(str(self.feature_pairs[idx][2])[2:-2])


            image1 = transforms.ToTensor()(image1)
            #print(image1.size())
            image2 = transforms.ToTensor()(image2)

            image = torch.stack((image1, image2))
            image_label = int(self.feature_pairs[idx][1])

            image_pair = [str(self.feature_pairs[idx][2])[2:-2] + ' && ' + str(self.feature_pairs[idx][3])[2:-2]]

            return image, image_label, image_pair
        def __len__(self):
            return self.feature_pairs.shape[0]

def get_train_loader_kinface2(type,i,COLLECTION,BATCH_SIZE,root):

    select_split = i
    test_indices = split[torch.arange(split.size(0)) == select_split - 1].view(-1).numpy().tolist()
    #print(test_indices)
    # print(split_set[str(select_split)][0])
    # print(split_set[str(select_split)][1])
    # print(split_set[str(select_split)][2])
    # print(split_set[str(select_split)][3])
    train_index = COLLECTION['split_' + str(split_set[str(select_split)][0])] \
                    + COLLECTION['split_' + str(split_set[str(select_split)][1])] \
                    + COLLECTION['split_' + str(split_set[str(select_split)][2])] \
                    + COLLECTION['split_' + str(split_set[str(select_split)][3])]

    train_index = train_index + train_index + train_index + train_index
    # print(train_index)
    print('train_index:', len(train_index))
    TRAIN_DATASET = kinface2_train_dataset(train_index,root)
    train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    return [train_loader,test_indices]
  


def get_train_loader_kinface1(type,i,F_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION,M_S_COLLECTION,BATCH_SIZE,root):
    
    select_split = i

    print(split_set[str(select_split)][0])
    print(split_set[str(select_split)][1])
    print(split_set[str(select_split)][2])
    print(split_set[str(select_split)][3])
    F_S_train_index =   F_S_COLLECTION['split_' + str(split_set[str(select_split)][0])] \
                    + F_S_COLLECTION['split_' + str(split_set[str(select_split)][1])] \
                    + F_S_COLLECTION['split_' + str(split_set[str(select_split)][2])] \
                    + F_S_COLLECTION['split_' + str(split_set[str(select_split)][3])]
    print('F_S_train_index:', len(F_S_train_index))
    F_D_train_index =   F_D_COLLECTION['split_' + str(split_set[str(select_split)][0])] \
                    + F_D_COLLECTION['split_' + str(split_set[str(select_split)][1])] \
                    + F_D_COLLECTION['split_' + str(split_set[str(select_split)][2])] \
                    + F_D_COLLECTION['split_' + str(split_set[str(select_split)][3])]
    print('F_D_train_index:', len(F_D_train_index))
    M_D_train_index =   M_D_COLLECTION['split_' + str(split_set[str(select_split)][0])] \
                    + M_D_COLLECTION['split_' + str(split_set[str(select_split)][1])] \
                    + M_D_COLLECTION['split_' + str(split_set[str(select_split)][2])] \
                    + M_D_COLLECTION['split_' + str(split_set[str(select_split)][3])]
    print('M_D_train_index:', len(M_D_train_index))
    M_S_train_index =   M_S_COLLECTION['split_' + str(split_set[str(select_split)][0])] \
                    + M_S_COLLECTION['split_' + str(split_set[str(select_split)][1])] \
                    + M_S_COLLECTION['split_' + str(split_set[str(select_split)][2])] \
                    + M_S_COLLECTION['split_' + str(split_set[str(select_split)][3])]
    print('M_S_train_index:', len(M_S_train_index))


    train_index = F_S_train_index + M_S_train_index + F_D_train_index + M_D_train_index

    print('train_index:', len(train_index))
    TRAIN_DATASET = kinface1_train_dataset(train_index,len(F_S_train_index), len(M_S_train_index), len(F_D_train_index), len(M_D_train_index ),root)
    train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    return [train_loader]

def make_test_indices(feature, split):
            indices = []
            for i in range(feature.shape[0]):
                if int(feature[i][0]) == int(split):
                    indices.append(i)
            return indices

   