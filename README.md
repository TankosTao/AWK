
# ATTENTION

We setup a password for the source codes, since the related paper is still in revision. please send email to huangsheng@cqu.edu.cn for requesting the access password of source codes.






# INTRODUCTION

KINSHIP-VERIFICATION


Inspired by human visual systems
that incorporate both low-order and high-order cross pair information from local and global perspectives, we propose to leverage
high-order cross-pair features and develop a novel end-to-end
deep learning model named the Adaptively Weighted k-Tuple
Metric Network (AWk-TMN).

 First, a novel cross-pair metric learning loss based on k-tuplet loss is introduced. It naturally captures both the low-order
and high-order discriminative features from multiple negative
pairs. Second, an adaptively weighting scheme is formulated to
better highlight hard negative examples among multiple negative
pairs, leading to enhanced performance. Third, different levels
of convolutional features are integrated to further exploit the
low-order and high-order representational power, with jointly
optimized feature and metric learning. Extensive experimental
results on three popular kinship verification datasets demonstrate
the effectiveness of the proposed AWk-TMN approach when
comparing with the state-of-the-art approaches.

# DATASET


# PYTORCH
```
conda create --n kinship python=3.7
conda activate kinship

# install dependencies

pip install -r requirements.txt

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch


``` 


## KINFACEW-I KINFACEW-II TSKINFACE
![Image text](../images_1/father-dau/fd_001_1.jpg)![Image text](../images_1/father-dau/fd_001_2.jpg) ![Image text](../images_1/mother-dau/md_001_1.jpg)![Image text](../images_1/mother-dau/md_001_2.jpg) ![Image text](../images_1/mother-son/ms_001_1.jpg)![Image text](../images_1/mother-son/ms_001_2.jpg) ![Image text](../images_1/father-son/fs_001_1.jpg)![Image text](../images_1/father-son/fs_001_2.jpg)

![Image text](../images_2/images/fd_001_1.jpg)![Image text](../images_2/images/fd_001_2.jpg) ![Image text](../images_2/images/fs_001_1.jpg)![Image text](../images_2/images/fs_001_2.jpg) ![Image text](../images_2/images/ms_001_1.jpg)![Image text](../images_2/images/ms_001_2.jpg) ![Image text](../images_2/images/md_001_1.jpg)![Image text](../images_2/images/md_001_2.jpg)

![Image text](../tskimg/md/md_1_d.jpg)![Image text](../tskimg/md/md_1_m.jpg) ![Image text](../tskimg/ms/ms_1_s.jpg)![Image text](../tskimg/ms/ms_1_m.jpg) ![Image text](../tskimg/fd/fd_1_d.jpg)![Image text](../tskimg/fd/fd_1_f.jpg) ![Image text](../tskimg/fs/fs_1_s.jpg)![Image text](../tskimg/fs/fs_1_f.jpg)



## ENVIRONMENT
 - pip install -r requirements.txt


## QUICK START

1. modify config file(configs/configs.yml) to point to the data set
2. type the following code

#### train 


```
python kinship.py --conv=conv1234 --dataset=KINFACE1 

```  

### eval

```
python eval/kinship_eval.py --conv=conv1234 --dataset=KINFACE1 --feature_encoder='yourmodelpath' --relation_network='yourmodelpath'
```  
### test your own  face images

```  
python test/kinship_test.py --conv=conv1234 --feature_encoder='yourmodelpath' --relation_network='yourmodelpath' --img1='the first img path of the pair-imgs you wanna test' --img2='the second img path of the pair-imgs you wanna test'
```  

<!-- ## EXPERIMENT
![father ]()


# EXTRA
### Add face detection algorithm from https://github.com/yeyupiaoling/Pytorch-MTCNN.git 

1. origin img pair


![father ](./dataset/2.jpg) ![son](./dataset/2_s.jpg) 

2. detect faces


![father ](./dataset/2_f_d.jpg) ![son](./dataset/2_s_d.jpg)

3.  kinship verification 


![father ](./dataset/2_f_2_s.jpg) -->

<!-- 
```
python detect_relation_from_origin_pic.py --conv=conv1234 --dataset=KINFACE1 --feature_encoder='yourmodelpath' --relation_network='yourmodelpath' --img1='the first img path of the pair-imgs you wanna test' --img2='the second img path of the pair-imgs you wanna test'

```   -->





# Introduction to Key Documents
 - `kinship.py` you can train our network with this file
 - `eval/kinship_eval.py` you can eval our network with this file
 - `test/kinship_test.py` you can test our network with this file
 - `model` the pretrained weights of our network 
 - `dataset` kinfaceW-I kinfaceW-II tskinface datasets
 - `conv`   
 <!-- - `conv`  
 - `models_mtcnn` nets of face detection algorithm
 - `utils` functions required by face detection algorithm
 - `infer_path.py` 使用路径预测图像，检测图片上人脸的位置和关键的位置，并显示
 - `infer_camera.py` 预测图像程序，检测图片上人脸的位置和关键的位置实时显示 -->
<!-- 


## reference 

1. https://github.com/yeyupiaoling/Pytorch-MTCNN.git  -->