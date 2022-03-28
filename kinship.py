from genericpath import isdir
from posixpath import pardir
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
from torch.functional import Tensor

####

parser = argparse.ArgumentParser(description="kinship Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-e","--episode",type = int, default= 5000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("--config",type = str,default='./configs/configs.yml')
parser.add_argument("-g","--gpu",type=int, default=0)

parser.add_argument("-s","--select_split",type = list, default=[1,2,3,4,5])
parser.add_argument("-k","--k_neg_pair",type = list, default=[2])

parser.add_argument("--alp",type = list, default=[0.6])
parser.add_argument("--margin",type = list, default=[0.4])
# Hyper Parameters
# a =[0.3,0.3,0.4,0.4,0.4,0.4]
# m =[0.1,0.2,0.1,0.2,0.3,0.4]
# a =[0.6,0.6,0.6,0.6,0.7,0.7,0.7]
# m =[0.1,0.2,0.3,0.4,0.1,0.2,0.3]
#a = [0.5]
#m = [0.5]
# a = [0.5,0.5,0.5,0.5,0.5]
# m = [0.1,0.2,0.3,0.4,0.5]

parser.add_argument("-t","--dataset",type = str, default='TSKINFACE',help='TSKINFACE or KINFACE1 or KINFACE2 or txt')
parser.add_argument("--conv",type = str, default='conv1234',help='conv14 or conv1234')
parser.add_argument("--eval_each_episode",type = int,default=50)
parser.add_argument("--times",type = int,default=5,help='repetition times for Parameters test')
parser.add_argument("--result_save_path",type = str, default='./tsk_test')
parser.add_argument("--is_save_weights",type = bool, default=True)
parser.add_argument("--addtion",type = str, default="TSKINFACE_conv1234_awk",help='description for this training')
parser.add_argument("--modelpath",type = str, default="./tsk_model")

parser.add_argument("--loss",type = str, default= 'awk',help='awk or triplet or k_tuple')
parser.add_argument("--is_save_log",type = bool, default= False,help='')
parser.add_argument("--log_path",type = str, default= 'log',help='./')


parser.add_argument("--pretrained",type = bool, default=False)
parser.add_argument("--feature_encoder",type = str, default="./1/feature_encoder_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_1_a_0.5_m_0.5.pkl")
parser.add_argument("--relation_network",type = str, default="./1/relation_network_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_1_a_0.5_m_0.5.pkl")
parser.add_argument("-d","--dataroot",type = str)
parser.add_argument("-p","--picroot",type = str)

args = parser.parse_args()

if args.dataset=='TSKINFACE':
    from data.loader_tskinface import get_train_loader,split_set,tskinface_test_dataset
elif args.dataset =='KINFACE1' or args.dataset == 'KINFACE2':
    from data.loader_kinface import load_kinface,get_train_loader_kinface2,get_train_loader_kinface1,make_test_indices,kinface_test_dataset
elif args.dataset == 'txt':
    from data.loader_from_txt import get_train_loader_txt
    


configs = yaml.load(open(args.config, 'rb'), Loader=yaml.Loader)

if args.dataset == 'TSKINFACE' or args.dataset=='KINFACE1'or args.dataset=='KINFACE2':
    args.dataroot = configs[args.dataset]['dataroot']
    args.picroot = configs[args.dataset]['picroot']




# imgdic = {}


# if args.memory_mode:
#     for path_1 in os.listdir(args.picroot):
        
#         if  os.path.isdir(os.path.join(args.picroot,path_1)):
            
#             if path_1 not in imgdic[args.picroot].keys():
#                 imgdic[args.picroot].update({path_1:{}})
#             #print(imgdic)
#             for path_2 in os.listdir(os.path.join(args.picroot,path_1)):
#                 if not path_2.endswith(".jpg"):
#                     continue
#                 if path_2 not in imgdic[args.picroot][path_1].keys():
#                     imgdic[args.picroot][path_1].update({path_2:cv.imread(os.path.join(args.picroot,path_1,path_2))})


if not os.path.exists(args.modelpath):
    os.mkdir(args.modelpath)
if not os.path.exists(args.result_save_path):
    os.mkdir(args.result_save_path)

if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)
# Hyper Parameters
# a =[0.3,0.3,0.4,0.4,0.4,0.4]
# m =[0.1,0.2,0.1,0.2,0.3,0.4]
# a =[0.6,0.6,0.6,0.6,0.7,0.7,0.7]
# m =[0.1,0.2,0.3,0.4,0.1,0.2,0.3]
a = args.alp
m = args.margin
# a = [0.5,0.5,0.5,0.5,0.5]
# m = [0.1,0.2,0.3,0.4,0.5]
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


if args.dataset == 'KINFACE2':
    feature_F_D,feature_F_S,feature_M_D,feature_M_S,COLLECTION = load_kinface(args.dataset,args.dataroot)
if args.dataset == 'KINFACE1':
    feature_F_D,feature_F_S,feature_M_D,feature_M_S,F_S_COLLECTION,M_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION = load_kinface(args.dataset,args.dataroot)







def main(K_PAIR,times):
    # step 1: init dataset
    for alp,margin in zip(a,m):
        for circle in SELECT_SPLIT:

            save_file_name = "{}_dataset_{}_k_{}_a_{}_m_{}_loss_{}_conv_{}".format(args.addtion,args.dataset,str(K_PAIR),str(alp),str(margin),args.loss,args.conv)
            #args.addtion+"_"+str(args.dataset)+"_"+str(K_PAIR)+"_a_"+str(alp)+"_m_"+str(margin)+".txt"
            if args.is_save_log:
                from utils.log import get_logger 
                logger = get_logger(log_file=os.path.join(args.log_path,save_file_name+".log"))
           
    
            if args.dataset == 'KINFACE1':
                related_list = get_train_loader_kinface1(args.dataset,circle,F_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION,M_S_COLLECTION,BATCH_SIZE,args.picroot)
                train_loader = related_list[0]
               
            elif args.dataset == 'KINFACE2':
                related_list = get_train_loader_kinface2(args.dataset,circle,COLLECTION,BATCH_SIZE,args.picroot)
                train_loader = related_list[0]
                test_indices = related_list[1]
                
            elif args.dataset == 'TSKINFACE':
                train_loader,test_index_for_son,test_index_for_dau,MAKE_test_index_for_fs,MAKE_test_index_for_ms,MAKE_test_index_for_fd,MAKE_test_index_for_md = get_train_loader(args.dataset,circle,split_set,BATCH_SIZE,args.picroot)
            
            elif args.dataset == 'txt':
                train_loader = get_train_loader_txt("./train/train.txt",BATCH_SIZE)
            
      

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
            last_correct_fs = []
            last_incorrect_fs = []
            last_correct_fd = []
            last_incorrect_fd = []
            last_correct_ms = []
            last_incorrect_ms = []
            last_correct_md = []
            last_incorrect_md = []

            for episode in range(EPISODE):
                
                feature_encoder.train()
                relation_network.train()
                feature_encoder_scheduler.step(episode)
                relation_network_scheduler.step(episode)
                # con
                # print(split)
                batch_features, batch_labels, batch_pairs = train_loader.__iter__().next()
                combine=[]
                parent_set, child_set = batch_features.split(split_size=1, dim=1)
                for remove_num in range(parent_set.size(0)):
                    res = list(range(parent_set.size(0)))
                    res.remove(remove_num)
                    combine.append(random.sample(res,K_PAIR))
                child_list = []
                for i in combine:
                    for j in i:
                        child_list.append(j)
                

                parent_features = Variable(parent_set.view(32, 3, 64, 64)).cuda(GPU).float()  # 32*1024
                child_features = Variable(child_set.view(32, 3, 64, 64)).cuda(GPU)  # k*312
                if args.conv == 'conv1234':
                    parent_features1, parent_features2, parent_features3, parent_features4 = feature_encoder(parent_features)
                    child_features1, child_features2, child_features3, child_features4 = feature_encoder(child_features)
                    # print(parent_features.size())

                    relation_pair1, relation_pair2,relation_pair3,relation_pair4 = conv_process_1234(parent_features1, parent_features2, parent_features3, parent_features4,child_features1, child_features2, child_features3, child_features4,child_list,K_PAIR,BATCH_SIZE, FEATURE_DIM,GPU)
                    relation = relation_network(relation_pair1, relation_pair2,relation_pair3,relation_pair4)
                if args.conv == 'conv14':
                    parent_features1, parent_features2 = feature_encoder(parent_features)
                    child_features1, child_features2 = feature_encoder(child_features)
                    relation_pair1, relation_pair2 = conv_process_14(parent_features1, parent_features2,child_features1, child_features2,child_list,K_PAIR,BATCH_SIZE, FEATURE_DIM,GPU)
                    relation = relation_network(relation_pair1, relation_pair2)

                # print(relation)
                relation_pos, relation_neg = relation[:32], relation[32:]
                relation_neg = relation_neg.view(32, K_PAIR)
                relation_matrix = torch.ones(32, K_PAIR).cuda(GPU)
                for i in range(BATCH_SIZE):
                    sum = 0
                    for j in range(K_PAIR):
                        relation_matrix[i][j] = relation_neg[i][j]
                if args.loss == 'awk':

             
                    loss_p = torch.sum(torch.clamp(torch.add(torch.div(relation_pos, -1), (alp+margin), out=None), min=0))
                    loss_n = torch.sum(torch.mul(relation_neg, torch.clamp(relation_neg - (alp-margin), min=0)))
                    
                    loss = loss_p + loss_n
                elif args.loss == 'triplet':
                    loss = torch.sum(torch.clamp(torch.max(relation_neg,dim=1)[0]-torch.max(relation_pos,dim=1)[0]+0.8,0))/32
                elif args.loss == 'k_tuple':
                    loss_p = torch.sum(torch.clamp(torch.add(torch.div(relation_pos, -1), 1.0, out=None), min=0))
                    loss_n = torch.sum(torch.mul(relation_matrix, torch.clamp(relation_neg - 0.2, min=0)))
                    loss = loss_p + loss_n

                # training

                feature_encoder.zero_grad()
                relation_network.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

                feature_encoder_optim.step()
                relation_network_optim.step()

                if (episode) % args.eval_each_episode == 0:
                    if not args.is_save_log:
                        print("episode:", episode, "loss", loss.data)
                        print("Parameters a:{:.1f},m:{:.1f},loss:{},conv:{},dataset:{},split:{:d},k_neg_pair:{:d},learning_rate:{:f}".format(alp,margin,args.loss,args.conv,args.dataset,circle,K_PAIR,LEARNING_RATE))
                  

                    # print("Parameters a:{:.1f},m:{:.1f},loss:{},conv:{},dataset:{},split:{:d},k_neg_pair:{:d},learning_rate:{:f}".format(alp,margin,args.loss,args.conv,args.dataset,circle,K_PAIR,LEARNING_RATE))
                    print("Parameters a_list:{},m_list:{},k_list:{}".format(str(a),str(m),str(args.k_neg_pair)))
                    # if args.is_save_log:
                    #     logger.info("episode:", episode, "loss", loss.data)
                    #     logger.info("Parameters a:{:.1f},m:{:.1f},loss:{},conv:{},dataset:{},split:{:d},k_neg_pair:{:d},learning_rate:{:f}".format(alp,margin,args.loss,args.conv,args.dataset,circle,K_PAIR,LEARNING_RATE))
                    # else:
                    # print("episode:", episode, "loss", loss.data)
                    # print("Parameters a:{:.1f},m:{:.1f},loss:{},conv:{},dataset:{},split:{:d},k_neg_pair:{:d},learning_rate:{:f}".format(alp,margin,args.loss,args.conv,args.dataset,circle,K_PAIR,LEARNING_RATE))
                    #f.write("\tepisode:"+str(episode)+" loss:"+str(loss.data)+' time-cost:'+str(endtime-start)+"\n")
                if episode % args.eval_each_episode == 0:
            
                    st = time.perf_counter()

                    print("Testing...",str(episode))
                    feature_encoder.eval()
                    relation_network.eval()

                    if args.dataset == 'txt':
                        def compute_accuracy(test_file):
                            # print(test_index)
                            total_rewards = 0
                            counter = 0
                            correct_list = []
                            incorrect_list = []
                        
                
                            test_loader =get_train_loader_txt(test_file,BATCH_SIZE)
                            for features, test_labels,feature_pair in test_loader:
                                # print(features.size())
                                # print(test_labels.size())
                                #batch_size = test_labels.shape[0]
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

                               # print("########",Tensor([1]).cuda(GPU))
                                # print(predict_labels)
                                # print(test_labels[j])

                                
                                # print(int(test_labels[j]))
                                # print(int(predict_labels[j].item()))
                                rewards = []
                                for j in range(len(test_labels)):

                                    if int(test_labels[j]) == int(predict_labels[j].item()):

                                        # if test_labels[j]=='1':
                                        rewards.append(1)
                                        # else:
                                        #     rewards.append(0)

                                    else:
                                        # if test_labels[j]=='0':
                                        rewards.append(0)
                                        # else:
                                        #     rewards.append(0)
                                    
                                    # if test_labels[j]=='1':
                                    #     base_tensor = Tensor(1).cuda(GPU)
                                    # elif test_labels[j]=='0':
                                    #     base_tensor = Tensor(0).cuda(GPU)

                                    # if base_tensor == predict_labels[j]:
                                    #     rewards.append(1)
                                    # else:
                                    #     rewards.append(0)
                                #rewards = [1 if predict_labels[j] == Tensor(int(test_labels[j])).cuda(GPU) else 0 for j in range(BATCH_SIZE)]
                                        # print(predict_labels)

                    
                                # for idex in range(BATCH_SIZE):
                                #     if predict_labels[idex] == test_labels[idex].cuda(GPU):
                                #         correct_list.append(str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))
                                #     else :
                                #         incorrect_list.append(str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))


                                total_rewards += np.sum(rewards)
                                counter += len(test_labels)
                            accuracy = total_rewards / 1.0 / counter

                            return accuracy, correct_list, incorrect_list

                        # print("father -- son")

                        kinship_accuracy, correct, incorrect = compute_accuracy('./train/test.txt')
                      


                        Mean = kinship_accuracy
                        # end = time.perf_counter()
                        if not args.is_save_log:
                            print('Mean:', Mean)
                            print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                kinship_accuracy_for_MD))
                            print('best mean : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                    kinship_accuracy_for_MD))

                        if Mean > last_Mean:
                            # save networks
                            best_epoch = episode
                            last_Mean = Mean
                            if args.is_save_weights:
                                    print("save networks for best-episode:", episode)
                                    torch.save(feature_encoder.state_dict(),
                                        join(args.modelpath,"feature_encoder_"+args.dataset+"_"+str(times)+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(circle) +"_a_"+str(alp)+"_m_"+str(margin)+ ".pkl"))
                                    torch.save(relation_network.state_dict(),
                                       join(args.modelpath, "relation_network_"+args.dataset+"_"+str(times)+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(circle) +"_a_"+str(alp)+"_m_"+str(margin)+ ".pkl"))
                        
                    
                    if args.dataset == 'TSKINFACE':
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
                        # end = time.perf_counter()
                        if not args.is_save_log:
                            print('Mean:', Mean)
                            print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                kinship_accuracy_for_MD))
                            print('best mean : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                    kinship_accuracy_for_MD))

                        if Mean > last_Mean:
                            # save networks
                            best_epoch = episode
                            last_Mean = Mean
                            last_kinship_accuracy_for_FS, last_correct_fs, last_incorrect_fs = kinship_accuracy_for_FS, correct_fs, incorrect_fs
                            last_kinship_accuracy_for_FD, last_correct_fd, last_incorrect_fd = kinship_accuracy_for_FD, correct_fd, incorrect_fd
                            last_kinship_accuracy_for_MS, last_correct_ms, last_incorrect_ms = kinship_accuracy_for_MS, correct_ms, incorrect_ms
                            last_kinship_accuracy_for_MD, last_correct_md, last_incorrect_md = kinship_accuracy_for_MD, correct_md, incorrect_md
                            if args.is_save_weights:
                                    print("save networks for best-episode:", episode)
                                    torch.save(feature_encoder.state_dict(),
                                        join(args.modelpath,"feature_encoder_"+args.dataset+"_"+str(times)+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(circle) +"_a_"+str(alp)+"_m_"+str(margin)+ ".pkl"))
                                    torch.save(relation_network.state_dict(),
                                       join(args.modelpath, "relation_network_"+args.dataset+"_"+str(times)+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(circle) +"_a_"+str(alp)+"_m_"+str(margin)+ ".pkl"))
                            
                    if args.dataset == 'KINFACE1' or args.dataset=='KINFACE2':
                        
                        def compute_accuracy(feature):
                            total_rewards = 0
                            counter = 0
                            correct_list = []
                            incorrect_list = []
                           
                         
                            if args.dataset == 'KINFACE1':
                                
                                indices = make_test_indices(feature, circle)
                                feature = feature[indices]
                            if args.dataset == 'KINFACE2':
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
                        if not args.is_save_log:
                            print('Mean:', Mean)
                            print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                kinship_accuracy_for_MD))
                            print('best mean : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                kinship_accuracy_for_MD))

                        if Mean > last_Mean:
                            # save networks
                            best_epoch = episode
                            last_Mean = Mean
                            last_kinship_accuracy_for_FS, last_correct_fs, last_incorrect_fs = kinship_accuracy_for_FS, correct_fs, incorrect_fs
                            last_kinship_accuracy_for_FD, last_correct_fd, last_incorrect_fd = kinship_accuracy_for_FD, correct_fd, incorrect_fd
                            last_kinship_accuracy_for_MS, last_correct_ms, last_incorrect_ms = kinship_accuracy_for_MS, correct_ms, incorrect_ms
                            last_kinship_accuracy_for_MD, last_correct_md, last_incorrect_md = kinship_accuracy_for_MD, correct_md, incorrect_md
                            if args.is_save_weights:
                                    print("save networks for best-episode:", episode)
                                    torch.save(feature_encoder.state_dict(),
                                        join(args.modelpath,"feature_encoder_"+args.dataset+"_"+str(times)+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(circle) +"_a_"+str(alp)+"_m_"+str(margin)+ ".pkl"))
                                    torch.save(relation_network.state_dict(),
                                       join(args.modelpath, "relation_network_"+args.dataset+"_"+str(times)+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(circle) +"_a_"+str(alp)+"_m_"+str(margin)+ ".pkl"))
                    # if args.is_save_log:
                    if args.is_save_log:
                        logger.info("episode:{},loss,{}".format(str(episode),str(loss.data)))
                        logger.info("Parameters a:{:.1f},m:{:.1f},loss:{},conv:{},dataset:{},split:{:d},k_neg_pair:{:d},learning_rate:{:f}".format(alp,margin,args.loss,args.conv,args.dataset,circle,K_PAIR,LEARNING_RATE))
                        if args.dataset == 'txt':
                            print('Mean:', Mean)
                            print("best:",last_Mean)
                            logger.info('Mean :%.4f, best:%.4f' % (
                                Mean, last_Mean))
                        else:
                        
                            logger.info('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                kinship_accuracy_for_MD))
                            if  episode%(args.eval_each_episode*5)==0:
                                logger.info('best mean : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                                    kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                                    kinship_accuracy_for_MD))
             

            print('Mean:', last_Mean)
            if not os.path.exists(os.path.join(args.result_save_path,args.addtion+"_"+str(args.dataset)+"_"+str(K_PAIR)+"_a_"+str(alp)+"_m_"+str(margin)+".txt")):
                f = open(os.path.join(args.result_save_path,args.addtion+"_"+str(args.dataset)+"_"+str(K_PAIR)+"_a_"+str(alp)+"_m_"+str(margin)+".txt"),"w",encoding='utf-8')
                f.close()
            
        
            f = open(os.path.join(args.result_save_path,args.addtion+"_"+str(args.dataset)+"_"+str(K_PAIR)+"_a_"+str(alp)+"_m_"+str(margin)+".txt"),"a",encoding='utf-8')
            if args.dataset=='txt':
                dic = {'time': time.asctime( time.localtime(time.time())),"Mean":last_Mean}
                

            else:
                print('Mean:', last_Mean)
                print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                    last_kinship_accuracy_for_FS, last_kinship_accuracy_for_FD, last_kinship_accuracy_for_MS,
                    last_kinship_accuracy_for_MD))
                dic = {'time': time.asctime( time.localtime(time.time())),"Mean":last_Mean,"split":circle,'try_time':times,"FS":last_kinship_accuracy_for_FS,'FD':last_kinship_accuracy_for_FD,'MS':last_kinship_accuracy_for_MS,"MD":last_kinship_accuracy_for_MD,'pkl':join(args.modelpath,args.dataset+"_"+str(times)+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(circle) +"_a_"+str(alp)+"_m_"+str(margin)+ ".pkl")}
                
            
            f.write(json.dumps(dic)+"\n")       
            f.close() 

           


               



if __name__ == '__main__':
    for K_PAIR in args.k_neg_pair:
        for times in range(args.times):
            main(K_PAIR,times)

