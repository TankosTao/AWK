import torch
from torch._C import default_generator
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
import time
from os.path import join
import json
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from data.loader_tskinface import get_train_loader,split_set,tskinface_test_dataset
from data.loader_kinface import load_kinface,get_train_loader_kinface2,get_train_loader_kinface1,make_test_indices,kinface_test_dataset




parser = argparse.ArgumentParser(description="kinship Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-e","--episode",type = int, default= 5000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-s","--select_split",type = list, default=1,help='1,2,3,4,5')
parser.add_argument("-k","--k_neg_pair",type = list, default=2)
parser.add_argument("--config",type = str,default='./configs/configs.yml')
parser.add_argument("-t","--type",type = str, default='KINFACE1',help='TSKINFACE or KINFACE1 or KINFACE2')
parser.add_argument("--conv",type = str, default='conv1234',help='conv14 or conv1234')
parser.add_argument("--pretrained",type = bool, default=True)
parser.add_argument("--feature_encoder",type = str, default="/home/hetao/KinFaceW-II/eval/2/feature_encoder_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_2_a_0.5_m_0.5.pkl")
parser.add_argument("--relation_network",type = str, default="/home/hetao/KinFaceW-II/eval/2/relation_network_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_2_a_0.5_m_0.5.pkl")
args = parser.parse_args()


configs = yaml.load(open(args.config, 'rb'), Loader=yaml.Loader)

args.dataroot = configs[args.type]['dataroot']
args.picroot = configs[args.type]['picroot']



# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
BATCH_SIZE = args.batch_size
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
SELECT_SPLIT = args.select_split
# K_PAIR = args.k_neg_pair

if args.conv == 'conv1234':
    from conv.related_net import CNNEncoder_1234 as  CNNEncoder
    from conv.related_net import RelationNetwork_1234 as  RelationNetwork
    from conv.related_op import conv_process_1234

if args.conv == 'conv14':
    from conv.related_net import CNNEncoder_14 as  CNNEncoder
    from conv.related_net import RelationNetwork_14 as  RelationNetwork
    from conv.related_op import conv_process_14


if args.type == 'KINFACE2':
    feature_F_D,feature_F_S,feature_M_D,feature_M_S,COLLECTION = load_kinface(args.type,args.dataroot)
if args.type == 'KINFACE1':
    feature_F_D,feature_F_S,feature_M_D,feature_M_S,F_S_COLLECTION,M_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION = load_kinface(args.type,args.dataroot)



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



def main():

        
    if args.type == 'KINFACE1':
        related_list = get_train_loader_kinface1(args.type,SELECT_SPLIT,F_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION,M_S_COLLECTION,BATCH_SIZE,args.picroot)
        train_loader = related_list[0]
        
    elif args.type == 'KINFACE2':
        related_list = get_train_loader_kinface2(args.type,SELECT_SPLIT,COLLECTION,BATCH_SIZE,args.picroot)
        train_loader = related_list[0]
        test_indices = related_list[1]
        
    elif args.type == 'TSKINFACE':
        test_index_for_son = FMS_COLLECTION['split_'+str(SELECT_SPLIT)]
        test_index_for_dau = FMD_COLLECTION['split_'+str(SELECT_SPLIT)]
        # MAKE_test_index_for_fs = make_pair(test_index_for_son)
        # print('MAKE_test_index_for_fs:', len(MAKE_test_index_for_fs))
        # MAKE_test_index_for_ms = make_pair(test_index_for_son)
        # print('MAKE_test_index_for_ms:', len(MAKE_test_index_for_ms))
        # MAKE_test_index_for_fd = make_pair(test_index_for_dau)
        # print('MAKE_test_index_for_fd:', len(MAKE_test_index_for_fd))
        # MAKE_test_index_for_md = make_pair(test_index_for_dau)
        # print('MAKE_test_index_for_md:', len(MAKE_test_index_for_md))
            
        MAKE_test_index_for_fs = list(reversed(test_index_for_son))
        print('MAKE_test_index_for_fs:', len(MAKE_test_index_for_fs))
        MAKE_test_index_for_ms = list(reversed(test_index_for_son))
        print('MAKE_test_index_for_ms:', len(MAKE_test_index_for_ms))
        MAKE_test_index_for_fd = list(reversed(test_index_for_dau))
        print('MAKE_test_index_for_fd:', len(MAKE_test_index_for_fd))
        MAKE_test_index_for_md = list(reversed(test_index_for_dau))
        print('MAKE_test_index_for_md:', len(MAKE_test_index_for_md))
    


    # init network
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)
    if args.pretrained:
        feature_encoder.load_state_dict(torch.load(args.feature_encoder))
        relation_network.load_state_dict(torch.load(args.relation_network))




    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)


    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=2000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=2000, gamma=0.5)

    print("training...")
    last_Mean = 0.0
    best_epoch = 0
    last_kinship_accuracy_for_FS = 0.0
    last_kinship_accuracy_for_FD = 0.0
    last_kinship_accuracy_for_MS = 0.0
    last_kinship_accuracy_for_MD = 0.0


    

    print("Testing...")
    feature_encoder.eval()
    relation_network.eval()
    
    if args.type == 'TSKINFACE':
        def compute_accuracy(MAKE_test_index, test_index,leirong):
            # print(test_index)
            total_rewards = 0
            counter = 0
            correct_list = []
            incorrect_list = []
            

            MAKE_test_index = MAKE_test_index + test_index
            test_index = test_index + test_index
            # print(MAKE_test_index)
            # print(test_index)
            # print(len(MAKE_test_index),len(test_index))
            test_dataset = tskinface_test_dataset(test_index, MAKE_test_index,len(test_index_for_son),len(test_index_for_dau),args.picroot,leirong)

            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
            for features, test_labels,feature_pair in test_loader:
                # print(features.size())
                # print(test_labels.size())
                batch_size = test_labels.shape[0]
                parent_set, child_set = features.split(split_size=1, dim=1)
                # print(sample_SD.reshape(32,6272).shape)
                # print(support_set.size())
                parent_feature = Variable(parent_set.view(-1, 3, 64, 64)).cuda(GPU).float()  # 32*1024
                child_feature = Variable(child_set.view(-1, 3, 64, 64)).cuda(GPU)  # k*312
                if args.conv == 'conv1234':
                    parent_feature1, parent_feature2, parent_feature3, parent_feature4 = feature_encoder(parent_feature)
                    child_feature1, child_feature2, child_feature3, child_feature4 = feature_encoder(child_feature)
                    # print(batch_features.size())
                    # print(sample_features.size())

                    relation_pairs1 = torch.cat((parent_feature1, child_feature1), 1).view(-1, FEATURE_DIM * 2, 32, 32)
                    relation_pairs2 = torch.cat((parent_feature2, child_feature2), 1).view(-1, FEATURE_DIM * 2, 16, 16)
                    relation_pairs3 = torch.cat((parent_feature3, child_feature3), 1).view(-1, FEATURE_DIM * 2, 16, 16)
                    relation_pairs4 = torch.cat((parent_feature4, child_feature4), 1).view(-1, FEATURE_DIM * 2, 14, 14)
                
                    # print(relation_pairs.size())

                    relations = relation_network(relation_pairs1, relation_pairs2, relation_pairs3, relation_pairs4).view(-1)
                if args.conv == 'conv14':
                    parent_feature1, parent_feature2 = feature_encoder(parent_feature)
                    child_feature1, child_feature2 = feature_encoder(child_feature)
                    # print(batch_features.size())
                    # print(sample_features.size())

                    relation_pairs1 = torch.cat((child_feature1, parent_feature1), 1).view(-1, FEATURE_DIM * 2,
                                                                                        32,
                                                                                        32)
                    relation_pairs2 = torch.cat((child_feature2, parent_feature2), 1).view(-1, FEATURE_DIM * 2,
                                                                                        14,
                                                                                        14)   

                    relations = relation_network(relation_pairs1, relation_pairs2).view(-1)

                    

                predict_labels = torch.gt(relations.data, 0.5).long()
                # print(predict_labels)

                rewards = [1 if predict_labels[j] == test_labels[j].cuda(GPU) else 0 for j in range(batch_size)]
                        # print(predict_labels)

                # rewards = [1 if predict_labels[j] == test_labels[j].cuda(GPU) else 0 for j in range(batch_size)]
                # print(predict_labels)
                # print(test_labels)
                for idex in range(batch_size):
                    if predict_labels[idex] == test_labels[idex].cuda(GPU):
                        correct_list.append(str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))
                    else :
                        incorrect_list.append(str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))


                total_rewards += np.sum(rewards)
                counter += batch_size
            accuracy = total_rewards / 1.0 / counter

            return accuracy, correct_list, incorrect_list

        # print("father -- son")

        kinship_accuracy_for_FS, correct_fs, incorrect_fs = compute_accuracy(MAKE_test_index_for_fs, test_index_for_son,'fs')
        # print("father -- daughter")
        kinship_accuracy_for_FD, correct_fd, incorrect_fd = compute_accuracy(MAKE_test_index_for_fd, test_index_for_dau,'fd')
        # print("mother -- son")
        kinship_accuracy_for_MS, correct_ms, incorrect_ms = compute_accuracy(MAKE_test_index_for_ms, test_index_for_son,'ms')
        # print("mother -- daughter")
        kinship_accuracy_for_MD, correct_md, incorrect_md = compute_accuracy(MAKE_test_index_for_md, test_index_for_dau,'md')


        Mean = (kinship_accuracy_for_FS + kinship_accuracy_for_FD + kinship_accuracy_for_MS + kinship_accuracy_for_MD) / 4.0
        end = time.perf_counter()
        print('Mean:', Mean)
        print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
            kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS, kinship_accuracy_for_MD))
    # f.write("epoch:"+str(episode)+" test-cost:"+str(end-st)+' kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
    # kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS, kinship_accuracy_for_MD)+" mean:"+str(Mean)+"\n")
            
    else:
        
        def compute_accuracy(feature):
            total_rewards = 0
            counter = 0
            correct_list = []
            incorrect_list = []
            
            
            if args.type == 'KINFACE1':
                indices = make_test_indices(feature, SELECT_SPLIT)
                feature = feature[indices]
            if args.type == 'KINFACE2':
                feature = feature[test_indices]
            # print(feature.shape)
            
            test_root = args.picroot

            TEST_DATASET = kinface_test_dataset(feature, test_root)
            test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=True)
            for features, test_labels, feature_pair in test_loader:
                batch_size = test_labels.shape[0]
                parent_set, child_set = features.split(split_size=1, dim=1)
                # print(sample_SD.reshape(32,6272).shape)
                # print(support_set.size())
                parent_feature = Variable(parent_set.view(-1, 3, 64, 64)).cuda(GPU).float()  # 32*1024
                child_feature = Variable(child_set.view(-1, 3, 64, 64)).cuda(GPU)  # k*312
                if args.conv == 'conv14':
                    parent_feature1, parent_feature2 = feature_encoder(parent_feature)
                    child_feature1, child_feature2 = feature_encoder(child_feature)
                    # print(batch_features.size())
                    # print(sample_features.size())

                    relation_pairs1 = torch.cat((child_feature1, parent_feature1), 1).view(-1, FEATURE_DIM * 2,
                                                                                        32,
                                                                                        32)
                    relation_pairs2 = torch.cat((child_feature2, parent_feature2), 1).view(-1, FEATURE_DIM * 2,
                                                                                        14,
                                                                                        14)
                    relations = relation_network(relation_pairs1, relation_pairs2).view(-1)
                if args.conv == 'conv1234':
                    parent_feature1, parent_feature2, parent_feature3, parent_feature4 = feature_encoder(parent_feature)
                    child_feature1, child_feature2, child_feature3, child_feature4 = feature_encoder(child_feature)
                    # print(batch_features.size())
                    # print(sample_features.size())

                    relation_pairs1 = torch.cat((parent_feature1, child_feature1), 1).view(-1, FEATURE_DIM * 2, 32, 32)
                    relation_pairs2 = torch.cat((parent_feature2, child_feature2), 1).view(-1, FEATURE_DIM * 2, 16, 16)
                    relation_pairs3 = torch.cat((parent_feature3, child_feature3), 1).view(-1, FEATURE_DIM * 2, 16, 16)
                    relation_pairs4 = torch.cat((parent_feature4, child_feature4), 1).view(-1, FEATURE_DIM * 2, 14, 14)
                
                    # print(relation_pairs.size())

                    relations = relation_network(relation_pairs1, relation_pairs2, relation_pairs3, relation_pairs4).view(-1)


                # print(relation_pairs.size())

                #relations = relation_network(relation_pairs1, relation_pairs2).view(-1)

                predict_labels = torch.gt(relations.data, 0.5).long()
                # print(predict_labels)

                rewards = [1 if predict_labels[j] == test_labels[j].cuda(GPU) else 0 for j in
                            range(batch_size)]
                # print(predict_labels)
                # print(test_labels)
                for idex in range(batch_size):
                    if predict_labels[idex] == test_labels[idex].cuda(GPU):
                        correct_list.append(
                            str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))
                    else:
                        incorrect_list.append(
                            str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))

                total_rewards += np.sum(rewards)
                counter += batch_size
            accuracy = total_rewards / 1.0 / counter

            return accuracy, correct_list, incorrect_list

        # print("father -- son")
        kinship_accuracy_for_FS, correct_fs, incorrect_fs = compute_accuracy(feature_F_S)
        # print("father -- daughter")
        kinship_accuracy_for_FD, correct_fd, incorrect_fd = compute_accuracy(feature_F_D)
        # print("mother -- son")
        kinship_accuracy_for_MS, correct_ms, incorrect_ms = compute_accuracy(feature_M_S)
        # print("mother -- daughter")
        kinship_accuracy_for_MD, correct_md, incorrect_md = compute_accuracy(feature_M_D)

        Mean = (
                            kinship_accuracy_for_FS + kinship_accuracy_for_FD + kinship_accuracy_for_MS + kinship_accuracy_for_MD) / 4.0

        print('Mean:', Mean)
        print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
            kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
            kinship_accuracy_for_MD))



               





if __name__ == '__main__':
    
    main()

