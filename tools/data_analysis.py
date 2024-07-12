# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------
import sys
sys.path.append('../')
import os
import cv2
import random 
import shutil
import numpy as np
from tqdm import tqdm
import os.path as osp
# import palettable
import matplotlib.pyplot as plt
from collections import defaultdict

# from utils import read_gdal
import gdal

# to solve the problem of 'ERROR 1: PROJ: pj_obj_create: Open of /opt/conda/share/proj failed'
os.environ['PROJ_LIB'] = r'C:\Users\HTHT\anaconda3\envs\torch17\Library\share\proj'

Class2Id = {
    'background': 0,
    'budiling': 1,
    'road': 2,
    'forest': 3,
    'glass': 4,
    'farmland': 5,
    'water': 6,
    'bareland': 7,
    # 'mine': 8,
    # 'special': 9
}

Id2Class = {
    0: 'background',
    1: 'budiling',
    2: 'road',
    3: 'forest',
    4: 'glass',
    5: 'farmland',
    6: 'water',
    7: 'bareland',
    # 8: 'mine',
    # 9: 'special'
}


colors = [  # bgr color
    [0, 0, 0], # 'background',
    [0, 255, 255],  # 'budiling',
    [0, 128, 0],  # 'road',
    [0, 255, 0],  # 'forest',
    [0, 128, 0],  # 'glass',
    [0, 0, 255],  # 'farmland',
    [255, 0, 0],  # 'water',
    [128, 128, 0],  # 'bareland',
    # [128, 0, 128],  # 'mine',
    # [255, 255, 0],  # 'special',
]


def read_gdal(path):
    '''
        读取一个tiff图像
    :param path: 要读取的图像路径(包括后缀名)
    :type path: string
    :return im_data: 返回图像矩阵(h, w, c)
    :rtype im_data: numpy
    :return im_proj: 返回投影信息
    :rtype im_proj: ?
    :return im_geotrans: 返回坐标信息
    :rtype im_geotrans: ?
    '''
    image = gdal.Open(path)  # 打开该图像
    if image == None:
        print(path + "文件无法打开")
        return
    img_w = image.RasterXSize  # 栅格矩阵的列数
    img_h = image.RasterYSize  # 栅格矩阵的行数
    # im_bands = image.RasterCount  # 波段数
    im_proj = image.GetProjection()  # 获取投影信息
    im_geotrans = image.GetGeoTransform()  # 仿射矩阵
    im_data = image.ReadAsArray(0, 0, img_w, img_h)

    # 二值图一般是二维，需要添加一个维度
    if len(im_data.shape) == 2:
        im_data = im_data[np.newaxis, :, :]

    im_data = im_data.transpose((1, 2, 0))
    return im_data, im_proj, im_geotrans


def get_fid(dir, ext):
    return [f for f in os.listdir(dir) if f.endswith(ext)]


def unique_folder(data_dir):
    fids = get_fid(data_dir, '_label.tif')
    n_class = len(DataGenerator.cls_info.items()) + 1
    counts = np.zeros(n_class, int)
    for f in tqdm(fids):
        label = cv2.imread(osp.join(data_dir, f), cv2.IMREAD_GRAYSCALE)
        _count = np.bincount(label.flatten(), minlength=n_class)
        counts += _count

    total = np.sum(counts)
    ratios = counts / total

    print('{:<10} | {:<10} | {:<10}'.format('class', 'count', 'ratio(%)'))
    for i in range(n_class):
        print('{:<10d} | {:<10d} | {:<10f}'.format(i, counts[i], ratios[i]*100))


def compute_mean_std(data_dirs):

    paths = []
    for _dir in data_dirs:
        fids = get_fid(_dir, '.tif')
        paths.extend([osp.join(_dir, fid+'.tif') for fid in fids])
    fnum = len(paths)
    print("samples:", len(paths))

    images = []
    for p in tqdm(paths):
        img = cv2.imread(p, cv2.IMREAD_LOAD_GDAL)
        img = img[:, :, :, np.newaxis]
        images.append(img)
    images = np.concatenate(images, axis=3).astype(np.float32) / 255. 
    
    means, stdevs = [], []
    for i in tqdm(range(4)):
        pixels = images[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print('mean:', means)
    print('std:', stdevs)


def calculate_mean_std(data_dir):

    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tif')]
    print("samples:", len(filepaths))

    images = []
    for path in tqdm(filepaths):
        img, _, _ = read_gdal(path)
        channel_num = img.shape[2]
        img = img.reshape([-1, channel_num])
        images.append(img)
    images = np.concatenate(images, axis=0).astype(np.float32)

    means, stdevs = [], []
    for i in tqdm(range(channel_num)):
        pixels = images[:, i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print('mean:', means)
    print('std:', stdevs)




if __name__ == '__main__':
    # src_dir = '/home/obtai/workspace/BuildingExtraction/DATASET/v4/256/train/labels'
    # unique_folder(src_dir)
    # class | ratio( % )
    # background | 0.228309
    # budiling | 8.412600
    # road | 3.976995
    # forest | 55.065185
    # glass | 2.834115
    # farmland | 14.184958
    # water | 15.269011
    # bareland | 0.028826


    # src_dirs = ['/home/ioai/Desktop/USER887/TIANCHI/dataset/src/suichang_round1_train_210120/',
    #             '/home/ioai/Desktop/USER887/TIANCHI/dataset/src/suichang_round1_test_partA_210120/']
    # compute_mean_std(src_dirs)
    # mean: [0.11894047 0.12947237 0.10506935 0.50785508]
    # std: [0.06690406 0.07723667 0.06991268 0.1627165 ]
    # compute_mean_std_2(src_dirs)
    # mean: [0.11894194, 0.12947349, 0.1050701, 0.50788707]
    # std: [0.08124223, 0.09198588, 0.08354711, 0.20507027]

    # dataset = '/home/work/ioai/USER887/TIANCHI2021/dataset/src/suichang_round1_train_210120/'
    # dataset = '/home/obtai/workspace/BuildingExtraction/DATASET/v4/256/train'
    # out_dir = '/home/obtai/workspace/BuildingExtraction/DATASET/v4/256/train_check/'
    # check_class(dataset, out_dir)
    # plot_sample_proportion(dataset, out_dir)

    # dataset_dirs = ['/home/work/ioai/USER887/TIANCHI2021/dataset/suichang_round1_train_210120/',
    #                 '/home/work/ioai/USER887/TIANCHI2021/dataset/suichang_round2_train_210316/']
    # sample_proportion(dataset_dirs)

    dir = r"D:\workspace\water\data\sentinel2\labeled_20220705\images_202110"
    calculate_mean_std(dir)
    # unique_folder(dir)
