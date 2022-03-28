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




def conv_process_1234(parent_features1, parent_features2, parent_features3, parent_features4,child_features1, child_features2, child_features3, child_features4,child_list,K_PAIR,BATCH_SIZE, FEATURE_DIM,GPU):
    new_child_feature1 = torch.Tensor(np.array([child_features1[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature1 = new_child_feature1.cuda(GPU)

    parent_features_ext1 = parent_features1.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext1 = torch.transpose(parent_features_ext1, 0, 1)
    #print(parent_features_ext1.size())
    parent_features_ext1 = parent_features_ext1.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 32, 32)
    # print(parent_features_ext.size())

    relation_pairs_pos1 = torch.cat((parent_features1, child_features1), 1).view(-1, FEATURE_DIM * 2, 32,32)
    relation_pairs_neg1 = torch.cat((parent_features_ext1, new_child_feature1), 1).view(-1, FEATURE_DIM * 2, 32, 32)
    relation_pair1 = torch.cat((relation_pairs_pos1, relation_pairs_neg1), 0)
    # print(relation_pair1.size())

    new_child_feature2 = torch.Tensor(np.array([child_features2[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature2 = new_child_feature2.cuda(GPU)

    parent_features_ext2 = parent_features2.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext2 = torch.transpose(parent_features_ext2, 0, 1)
    #print(parent_features_ext2.size())
    parent_features_ext2 = parent_features_ext2.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 16, 16)
    # print(parent_features_ext.size())

    relation_pairs_pos2 = torch.cat((parent_features2, child_features2), 1).view(-1, FEATURE_DIM * 2, 16, 16)
    relation_pairs_neg2 = torch.cat((parent_features_ext2, new_child_feature2), 1).view(-1, FEATURE_DIM * 2, 16,
                                                                                        16)
    relation_pair2 = torch.cat((relation_pairs_pos2, relation_pairs_neg2), 0)

    new_child_feature3 = torch.Tensor(np.array([child_features3[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature3 = new_child_feature3.cuda(GPU)

    parent_features_ext3 = parent_features3.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext3 = torch.transpose(parent_features_ext3, 0, 1)
    # print(parent_features_ext1.size())
    parent_features_ext3 = parent_features_ext3.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 16, 16)
    # print(parent_features_ext.size())

    relation_pairs_pos3 = torch.cat((parent_features3, child_features3), 1).view(-1, FEATURE_DIM * 2, 16, 16)
    relation_pairs_neg3 = torch.cat((parent_features_ext3, new_child_feature3), 1).view(-1, FEATURE_DIM * 2, 16,
                                                                                        16)
    relation_pair3 = torch.cat((relation_pairs_pos3, relation_pairs_neg3), 0)
    # print(relation_pair1.size())

    new_child_feature4 = torch.Tensor(np.array([child_features4[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature4 = new_child_feature4.cuda(GPU)

    parent_features_ext4 = parent_features4.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext4 = torch.transpose(parent_features_ext4, 0, 1)
    # print(parent_features_ext1.size())
    parent_features_ext4 = parent_features_ext4.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 14, 14)
    # print(parent_features_ext.size())

    relation_pairs_pos4 = torch.cat((parent_features4, child_features4), 1).view(-1, FEATURE_DIM * 2, 14, 14)
    relation_pairs_neg4 = torch.cat((parent_features_ext4, new_child_feature4), 1).view(-1, FEATURE_DIM * 2, 14,
                                                                                        14)
    relation_pair4 = torch.cat((relation_pairs_pos4, relation_pairs_neg4), 0)
    return relation_pair1,relation_pair2,relation_pair3,relation_pair4


def conv_process_14(parent_features1, parent_features2,child_features1, child_features2,child_list,K_PAIR,BATCH_SIZE, FEATURE_DIM,GPU):
    new_child_feature1 = torch.Tensor(
                        np.array([child_features1[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature1 = new_child_feature1.cuda(GPU)
    

    parent_features_ext1 = parent_features1.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext1 = torch.transpose(parent_features_ext1, 0, 1)
    parent_features_ext1 = parent_features_ext1.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 32, 32)
    # print(parent_features_ext.size())

    relation_pairs_pos1 = torch.cat((parent_features1, child_features1), 1).view(-1, FEATURE_DIM * 2, 32,
                                                                                32)
    relation_pairs_neg1 = torch.cat((parent_features_ext1, new_child_feature1), 1).view(-1, FEATURE_DIM * 2,
                                                                                        32,
                                                                                        32)
    relation_pair1 = torch.cat((relation_pairs_pos1, relation_pairs_neg1), 0)
    # print(relation_pair1.size())

    new_child_feature2 = torch.Tensor(
        np.array([child_features2[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature2 = new_child_feature2.cuda(GPU)
# print("###################",new_child_feature2.shape)

    parent_features_ext2 = parent_features2.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext2 = torch.transpose(parent_features_ext2, 0, 1)
    parent_features_ext2 = parent_features_ext2.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 14, 14)
    # print(parent_features_ext.size())
# print(parent_features_ext2.shape)

    relation_pairs_pos2 = torch.cat((parent_features2, child_features2), 1).view(-1, FEATURE_DIM * 2, 14,
                                                                                14)
# print(parent_features_ext2.shape)
    relation_pairs_neg2 = torch.cat((parent_features_ext2, new_child_feature2), 1).view(-1, FEATURE_DIM * 2,
                                                                                        14,
                                                                                        14)
    relation_pair2 = torch.cat((relation_pairs_pos2, relation_pairs_neg2), 0)
    return relation_pair1,relation_pair2

