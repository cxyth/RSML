# -*- encoding: utf-8 -*-
# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------
import os
import sys
import cv2
import os.path as osp
import time as T
from glob import glob
from tqdm import tqdm
import numpy as np
from scipy import ndimage as ndi
from datasets import Sentinel2_Dataset
from utils import randering_mask
from utils import read_gdal, write_gdal, uint16_to_8
# from sklearn.externals import joblib  # scikit-learn <= 0.21
import joblib   # scikit-learn > 0.21
import multiprocessing as mp
from functools import partial


def NDWI(img_data):
    # img_data (N, 10)
    # NDWI: (G-NIR)/(G+NIR)
    G = img_data[..., 1].copy().astype(np.float32)
    NIR = img_data[..., 6].copy().astype(np.float32)
    # index = (G - NIR) / (G + NIR)     # 实际使用时影像中会有NoData区域（默认填充0），直接除会警告
    a = G - NIR
    b = G + NIR
    index = np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b != 0)
    index = np.expand_dims(index, axis=-1)
    return index

def NDVI(img_data):
    # img_data (N, 10)
    # NDVI: (NIR-R)/(NIR+R)
    R = img_data[..., 2].copy().astype(np.float32)
    NIR = img_data[..., 6].copy().astype(np.float32)
    # index = (NIR - R) / (NIR + R)     # 实际使用时影像中会有NoData区域（默认填充0），直接除会警告
    a = NIR - R
    b = NIR + R
    index = np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b != 0)
    index = np.expand_dims(index, axis=-1)
    return index


def func(d, m):
    return m.predict(d)


def predict_in_single_image_multiprocessing(model, image, num_worker=None):
    img_h, img_w, _ = image.shape
    pred = np.zeros((img_h, img_w), dtype=np.float32)
    worker = partial(func, m=model)
    batch_size = num_worker if num_worker is not None else mp.cpu_count()
    n = int(np.ceil(img_h / batch_size))
    for i in tqdm(range(n)):
        dh = batch_size * i
        dh2 = dh + batch_size if dh + batch_size <= img_h else img_h
        batch_data = image[dh:dh2]
        with mp.Pool(batch_size) as p:
            batch_pred = p.map(worker, list(batch_data))
        pred[dh:dh2] = np.array(batch_pred)
    return pred


if __name__ == '__main__':
    DATA_PATH = r"E:\Jiang\water\data\sentinel2\labeled_20220705\images_202110"    # 输入数据路径
    OUTPUT_PATH = r"./runs/ndvi_2021"     # 保存路径

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    TEST_SET = sorted([os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.tif')])
    print('Test image number:', len(TEST_SET))

    t0 = T.time()
    for fpath in TEST_SET:
        print('> processing:', fpath)
        fname = osp.split(fpath)[-1]
        image, im_proj, im_geotrans = read_gdal(fpath)
        img_h, img_w, band = image.shape

        # 预处理
        # ndwi = NDWI(image)
        ndvi = NDVI(image)

        save_path = osp.join(OUTPUT_PATH, fname)
        write_gdal(ndvi.reshape(img_h, img_w, 1), save_path, im_proj, im_geotrans)
        print('> write: ', save_path)

    t1 = T.time()
    print("处理时间(s)：{:.3f}".format(t1-t0))