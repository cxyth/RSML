# -*- encoding: utf-8 -*-
# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------
import os
import sys
import time

import cv2
import os.path as osp
import time as T
from glob import glob
from tqdm import tqdm
import numpy as np
from scipy import ndimage as ndi
from datasets import Landsat_Dataset
from utils import read_gdal, write_gdal
# from sklearn.externals import joblib  # scikit-learn <= 0.21
import joblib   # scikit-learn > 0.21
import multiprocessing as mp
from functools import partial

# intel的加速库
# from sklearnex import patch_sklearn
# patch_sklearn()


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


def predict_in_single_image(model, image):
    img_h, img_w, bands = image.shape
    input = image.reshape(-1, bands)
    output = model.predict(input)
    return output.reshape(img_h, img_w)


if __name__ == '__main__':

    MODEL_PATH = "./runs/ndvi/m00.model"
    # MODEL_PATH = "runs/rf.v1.pca_ndwi/m00.model"  # 模型路径
    PCA_PATH = "./runs/pca.v1/pca.model"
    DATA_PATH = "../datasets/v1/images"    # 输入数据路径
    OUTPUT_PATH = "./runs/ndvi/pred"     # 保存路径

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    TEST_SET = sorted([os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.tif')])
    print('Test image number:', len(TEST_SET))

    # 加载模型
    svm_rbf = joblib.load(MODEL_PATH)

    pca = joblib.load(PCA_PATH) if PCA_PATH else None

    t0 = T.time()
    for fpath in TEST_SET:
        print('> processing:', fpath)
        fname = osp.split(fpath)[-1]
        image, im_proj, im_geotrans = read_gdal(fpath)
        img_h, img_w, band = image.shape

        # 预处理
        image = image.reshape(-1, band)
        image = Landsat_Dataset.transform(image, pca=pca)
        image = image.reshape(img_h, img_w, image.shape[1])

        # 预测
        pred = predict_in_single_image(svm_rbf, image)

        # 后处理
        pred_mask = pred.astype(np.uint8) + 1

        mask_name = osp.join(OUTPUT_PATH, fname)
        write_gdal(pred_mask.reshape(img_h, img_w, 1), mask_name, im_proj, im_geotrans)
        print('> write: ', mask_name)

    t1 = T.time()
    print("处理时间(s)：{:.3f}".format(t1-t0))