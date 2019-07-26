# -*- coding=utf-8 -*-
from scipy import misc
import cv2
import numpy as np
import os
import shutil

def remove_dim(img,dim_threshold):
    if img is not None:
        if 3 == len(img.shape):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img, cv2.CV_64F).var()
        # print('=== dim score: ', imageVar)
        if imageVar < dim_threshold:
            return 0
        else:
            return 1


def remove_gray(img,grayscale_threshold):
    # single channel just deal as good img in web dataset
    if 2 == len(img.shape):
        print('------------------------------------------------')
        return 1

    else:
        layer1 = img[:, :, 0].reshape((img.shape[0] * img.shape[1],)).astype(np.float64)
        layer2 = img[:, :, 1].reshape((img.shape[0] * img.shape[1],)).astype(np.float64)
        layer3 = img[:, :, 2].reshape((img.shape[0] * img.shape[1],)).astype(np.float64)
        midlayer = (layer1 + layer2 + layer3) / 3.
        dis = (np.linalg.norm(layer1 - midlayer) + np.linalg.norm(layer2 - midlayer) + np.linalg.norm(
            layer2 - midlayer)) / 3
        # dis = (np.sum(np.abs(layer1-midlayer))+np.sum(np.abs(layer2-midlayer))+np.sum(np.abs(layer3-midlayer)))/3./len(layer1)
        print('=== gray dis  : ' + str(dis))
        if dis < grayscale_threshold and dis > 10:
            return  0
        else:
            return 1



def rm_main_backup(scaled,  dim_threshold_gray, dim_threshold_color, grayscale_threshold):

    # scaled = misc.imresize(scaled, (160, 160, 3), interp='bilinear')
    flag = remove_gray(scaled, grayscale_threshold)
    if  flag == 0:
        # misc.imsave(os.path.join(args.grayscale_image_dir, filename+'.png'), scaled)
        scaled = None
    else:
        dim_score = remove_dim(scaled, dim_threshold_gray, dim_threshold_color)
        if dim_score == 0:
            # misc.imsave(os.path.join(args.grayscale_image_dir, filename+'.png'), scaled)
            scaled = None
    return scaled


def rm_main(scaled,  dim_threshold):
    dim_score = remove_dim(scaled, dim_threshold)
    if dim_score == 0:
        # misc.imsave(os.path.join(args.grayscale_image_dir, filename+'.png'), scaled)
        return 1
    else:
        return 0



# if __name__ == '__main__':

#     # img = misc.imread('C:\\dataset\\imdb_crop\\01\\nm0000001_rm3343756032_1899-5-10_1970.jpg')
#     # img = misc.imread('C:\\dataset\\imdb_crop\\01\\nm0000401_rm172467456_1961-7-30_2013.jpg')

#     grayscale_threshold1 = 300 # 黑白阈值， 公开数据集，遇到单通道的黑白图直接算好的图片

#     dim_threshold = 30  # 模糊阈值

#     img_dir = 'D:\\Desktop\\kaiyu.zhong\\Desktop\\tmp\\tmp_bad'
#     img_dir_bad = 'D:\\Desktop\\kaiyu.zhong\\Desktop\\tmp\\tmp_bad1'
#     img_list = os.listdir(img_dir)
#     print (img_list)
#     for dir in img_list:
#         print (os.path.join(img_dir, dir))
#         img = misc.imread(os.path.join(img_dir, dir))

#         # img = main(img,  dim_threshold_gray, dim_threshold_color, grayscale_threshold1)
#         img = rm_main(img,  dim_threshold)

#         if img is None:
#             shutil.move(os.path.join(img_dir, dir),os.path.join(img_dir_bad, dir))
