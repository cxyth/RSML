# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------
import os
import sys
import time
import joblib   # scikit-learn > 0.21
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import metrics, preprocessing

from datasets import Landsat_Dataset
from utils import read_gdal


if __name__ == '__main__':
    classes = ["shaqiu", "xintan", "gengdi", "building", "water", "veg"]

    dataset_dir = r"C:\Users\Admin\Desktop\HJN\datasets\v1"
    logdir = osp.join(os.getcwd(), 'runs', 'pca.v1')
    os.makedirs(logdir, exist_ok=True)

    # load data
    dataset = Landsat_Dataset(dataset_dir=dataset_dir, alpha=0.3, random_seed=2024)
    x_train, y_train = dataset.get_train_set()
    x_val, y_val = dataset.get_val_set()

    label = np.concatenate([y_train, y_val], axis=0)
    data = np.concatenate([x_train, x_val], axis=0)
    # ndwi = dataset.NDWI(data)
    data = dataset.Normalize(data)

    print(type(label), type(data))
    print(label[:10])
    print(data[:10])

    pca = PCA(3)
    t0 = time.time()
    pca.fit(data)
    t1 = time.time()
    data_dr = pca.transform(data)
    print(data_dr[:10], data_dr.shape, data_dr.dtype)
    t2 = time.time()
    print('fit(s):', t1-t0, 'transform(s):', t2-t1)
    # 查看解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Cumulative Explained Variance Ratio:", explained_variance_ratio.cumsum())

    # plt.figure()
    # for i in range(2):
    #     plt.scatter(x=data_dr[label==i, 0], y=data_dr[label==i, 0], c=colors[i], label=classes[i])
    # plt.legend()
    # plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(classes)):
        _mask = label==i
        print(np.unique(_mask, return_counts=True))
        ax.scatter(xs=data_dr[_mask, 0], ys=data_dr[_mask, 1], zs=data_dr[_mask, 2], c=f'C{i}', label=classes[i], alpha=0.6)
    plt.legend()
    plt.show()

    # 保存
    joblib.dump(pca, osp.join(logdir, 'pca.model'))


