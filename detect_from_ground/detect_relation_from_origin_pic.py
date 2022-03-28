import argparse
import os

from torch.serialization import save

import cv2
import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from utils.utils import generate_bbox, py_nms, convert_to_square
from utils.utils import pad, calibrate_box, processed_image
from detect_from_ori_pic.test_tskinface import tskinface_test_dataset
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='infer_models',      help='PNet、RNet、ONet三个模型文件存在的文件夹路径')
parser.add_argument('--image_path', type=str, default='dataset/test.jpg',  help='需要预测图像的路径')

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
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
GPU = args.gpu


device = torch.device("cuda")

# 获取P模型
pnet = torch.jit.load(os.path.join(args.model_path, 'PNet.pth'))
pnet.to(device)
softmax_p = torch.nn.Softmax(dim=0)
pnet.eval()

# 获取R模型
rnet = torch.jit.load(os.path.join(args.model_path, 'RNet.pth'))
rnet.to(device)
softmax_r = torch.nn.Softmax(dim=-1)
rnet.eval()

# 获取R模型
onet = torch.jit.load(os.path.join(args.model_path, 'ONet.pth'))
onet.to(device)
softmax_o = torch.nn.Softmax(dim=-1)
onet.eval()

if args.conv == 'conv1234':
    from conv.related_net import CNNEncoder_1234 as  CNNEncoder
    from conv.related_net import RelationNetwork_1234 as  RelationNetwork
    from conv.related_op import conv_process_1234

if args.conv == 'conv14':
    from conv.related_net import CNNEncoder_14 as  CNNEncoder
    from conv.related_net import RelationNetwork_14 as  RelationNetwork
    from conv.related_op import conv_process_14

feature_encoder = CNNEncoder()
relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)
feature_encoder.load_state_dict(torch.load(args.feature_encoder))
relation_network.load_state_dict(torch.load(args.relation_network))


feature_encoder.cuda(GPU)
relation_network.cuda(GPU)
feature_encoder.eval()
relation_network.eval()







# 使用PNet模型预测
def predict_pnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用RNet模型预测
def predict_rnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用ONet模型预测
def predict_onet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()


# 获取PNet网络输出结果
def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T

    return boxes_c


# 获取RNet网络输出结果
def detect_rnet(im, dets, thresh):
    """通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
    """
    h, w, c = im.shape
    # 将pnet的box变成包含它的正方形，可以避免信息损失
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    # 调整超出图像的box
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    for i in range(int(num_boxes)):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except:
            continue
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None

    keep = py_nms(boxes, 0.4, mode='Union')
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c


# 获取ONet模型预测结果
def detect_onet(im, dets, thresh):
    """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 128
        cropped_ims[i, :, :, :] = img
    cls_scores, reg, landmark = predict_onet(cropped_ims)

    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None

    w = boxes[:, 2] - boxes[:, 0] + 1

    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    boxes_c = calibrate_box(boxes, reg)

    keep = py_nms(boxes_c, 0.6, mode='Minimum')
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark


# 预测图片
def infer_image(img_pair):
    cut_img_pair=[]
    ori_img_pair = []
    for image_path in img_pair:
        #print(image_path)
        im = cv2.imread(image_path)
        # 调用第一个模型预测
        boxes_c = detect_pnet(im, 20, 0.79, 0.9)
        if boxes_c is None:
            return None, None
        # 调用第二个模型预测
        boxes_c = detect_rnet(im, boxes_c, 0.6)
        if boxes_c is None:
            return None, None
        # 调用第三个模型预测
        boxes_c, landmark = detect_onet(im, boxes_c, 0.7)
        if boxes_c is None:
            return None, None
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # print(corpbbox)
            im = im[corpbbox[1]:corpbbox[3],corpbbox[0]:corpbbox[2]]
            ori_img_pair.append(im)
            im = cv2.resize(im,(64,64))
            cut_img_pair.append(im)
            break
            
    
        #cv2.imwrite(os.path.join("results",os.path.basename(image_path)),img)
        #print(len(cut_img_pair))
    res1 = np.hstack([cut_img_pair[0], cut_img_pair[1]])
    res2 = np.hstack([ori_img_pair[0], ori_img_pair[1]])
    # cv2.imwrite(os.path.join("results",'1.jpg'),res)
    return cut_img_pair,res1,res2
        #return boxes_c, landmark


# 画出人脸框和关键点
def draw_face(image_path,img_pair_list, boxes_c, landmarks):

    for i in img_pair_list:
        img1 = cv2.imread(i[0])
        img2 = cv2.imread(i[1])
        

    img = cv2.imread(image_path)
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # print(corpbbox)
        img = img[corpbbox[1]:corpbbox[3],corpbbox[0]:corpbbox[2]]
        img = cv2.resize(img,(64,64))
        
    
        cv2.imwrite(os.path.join("results",os.path.basename(image_path)),img)
        break

        # (corpbbox[0], corpbbox[1]) 左上角点
        # (corpbbox[2], corpbbox[3]) 右下角点
        # 画人脸框
        # cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
        #               (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        # # 判别为人脸的置信度
        # cv2.putText(img, '{:.2f}'.format(score),
        #             (corpbbox[0], corpbbox[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # # 画关键点
    # for i in range(landmarks.shape[0]):
    #     for j in range(len(landmarks[i]) // 2):
    #         cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
    
    

    # cv2.imshow('result', img)
    # cv2.waitKey(0)
def detect_faces(image_path):
    # 预测图片获取人脸的box和关键点
    boxes_c, landmarks = infer_image(image_path)
    # print(boxes_c)
    # print(landmarks)
    # 把关键画出来
    if boxes_c is not None:
        draw_face(image_path=image_path, boxes_c=boxes_c, landmarks=landmarks)
    else:
        print('image not have face')



if __name__ == '__main__':
    # 预测图片获取人脸的box和关键点
    img_pair_list = [[args.img1,args.img2]]
    cut_pairs = []
    results_img = []
    results = []
    for i in img_pair_list:
        cut_pair,res_img,res_img1 = infer_image(i)
        cut_pairs.append(cut_pair)
        results_img.append(res_img)
        #results_img 存放原图
    
    test_dataset = tskinface_test_dataset(cut_pairs)
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
            print('1','related')
            results.append("1 related")
        else:
            print('0','unrelated')
            results.append("0 unrelated")
    print(len(img_pair_list))
    for i in range(len(img_pair_list)):
        cv2.putText(results_img[i], results[i],
                    (5, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        save_name = os.path.basename(img_pair_list[i][0]).split(".")[0]+"_"+os.path.basename(img_pair_list[i][1]).split(".")[0]+".jpg"
        print(save_name)
        cv2.imwrite(os.path.join("results",save_name),results_img[i])

    
