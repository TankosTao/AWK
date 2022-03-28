import torch
from torch.functional import Tensor
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
from test_tskinface import tskinface_test_dataset 
from os.path import join
import sys
sys.path.append("..")




parser = argparse.ArgumentParser(description="kinship Recognition in KinFaceW - II")



parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-t","--type",type = str, default='TSKINFACE')

parser.add_argument("--conv",type = str, default="conv1234")
parser.add_argument("--feature_encoder",type = str, default='/home/hetao/KinFaceW-II/eval/2/feature_encoder_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_2_a_0.5_m_0.5.pkl')
parser.add_argument("--relation_network",type = str, default='/home/hetao/KinFaceW-II/eval/2/relation_network_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_2_a_0.5_m_0.5.pkl')
parser.add_argument("--img1",type = str, default='/home/hetao/KinFaceW-II/eval/2/relation_network_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_2_a_0.5_m_0.5.pkl')
parser.add_argument("--img2",type = str, default='/home/hetao/KinFaceW-II/eval/2/relation_network_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_2_a_0.5_m_0.5.pkl')

args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
GPU = args.gpu






if args.conv == 'conv1234':
    from conv.related_net import CNNEncoder_1234 as  CNNEncoder
    from conv.related_net import RelationNetwork_1234 as  RelationNetwork
    from conv.related_op import conv_process_1234

if args.conv == 'conv14':
    from conv.related_net import CNNEncoder_14 as  CNNEncoder
    from conv.related_net import RelationNetwork_14 as  RelationNetwork
    from conv.related_op import conv_process_14





def main(test_list):

    # step 1: init dataset

    
        
        
        import time
        start = time.perf_counter()
        
        # init network
        print("init neural networks")

        feature_encoder = CNNEncoder()
        relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)
        feature_encoder.load_state_dict(torch.load(args.feature_encoder))
        relation_network.load_state_dict(torch.load(args.relation_network))
        

        feature_encoder.cuda(GPU)
        relation_network.cuda(GPU)

        # feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
        # feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=2000, gamma=0.5)
        # relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
        # relation_network_scheduler = StepLR(relation_network_optim, step_size=2000, gamma=0.5)

    

        feature_encoder.eval()
        relation_network.eval()

        
        
        test_dataset = tskinface_test_dataset(test_list)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for features,pair in test_loader:
          
            parent_set, child_set = features.split(split_size=1, dim=1)
            parent_feature = Variable(parent_set.view(-1, 3, 64, 64)).cuda(GPU).float()  # 32*1024
            child_feature = Variable(child_set.view(-1, 3, 64, 64)).cuda(GPU)  # k*312

            if args.conv == 'conv1234':
                parent_feature1, parent_feature2, parent_feature3, parent_feature4 = feature_encoder(parent_feature)
                child_feature1, child_feature2, child_feature3, child_feature4 = feature_encoder(child_feature)
           
                relation_pairs1 = torch.cat((parent_feature1, child_feature1), 1).view(-1, FEATURE_DIM * 2, 32, 32)
                relation_pairs2 = torch.cat((parent_feature2, child_feature2), 1).view(-1, FEATURE_DIM * 2, 16, 16)
                relation_pairs3 = torch.cat((parent_feature3, child_feature3), 1).view(-1, FEATURE_DIM * 2, 16, 16)
                relation_pairs4 = torch.cat((parent_feature4, child_feature4), 1).view(-1, FEATURE_DIM * 2, 14, 14)
         
                relations = relation_network(relation_pairs1, relation_pairs2, relation_pairs3, relation_pairs4).view(-1)
            if args.conv == 'conv14':
                parent_feature1, parent_feature2 = feature_encoder(parent_feature)
                child_feature1, child_feature2 = feature_encoder(child_feature)
                

                relation_pairs1 = torch.cat((child_feature1, parent_feature1), 1).view(-1, FEATURE_DIM * 2,
                                                                                    32,
                                                                                    32)
                relation_pairs2 = torch.cat((child_feature2, parent_feature2), 1).view(-1, FEATURE_DIM * 2,
                                                                                    14,
                                                                                    14)   

                relations = relation_network(relation_pairs1, relation_pairs2).view(-1)
            
            # print(relations.data)

            predict_labels = torch.gt(relations.data, 0.5).long()
            
            if Tensor([1]).cuda(GPU) == predict_labels.data:
                print(pair,'1','related')
            else:
                print(pair,'0','unrelated')

            

    
                
                
               
                
                


        
     



if __name__ == '__main__':
    results = [[args.img1,args.img2]]
    # for j in range(1,10):
    #     results.append(['/home/hetao/KinFaceW-II/tskimg/fd/fd_'+str(j)+"_f.jpg",'/home/hetao/KinFaceW-II/tskimg/fd/fd_'+str(j)+"_d.jpg"])
    
    main(results)

