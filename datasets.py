# -*- encoding: utf-8 -*-
# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------
import os
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils import read_gdal, np_divide


#---------------------------------------------------
#   从栅格数据构建数据集
#---------------------------------------------------
class Landsat_Dataset(object):
    # 使用data_analysis/data_analysis.py的calculate_mean_std()计算得到
    mean = np.array([961.8782, 1053.7188, 1302.2211, 1573.1169, 2185.5655, 2097.9481, 1643.7379])
    std = np.array([362.9614, 395.8249, 512.3208, 599.7623, 769.1146, 734.0038, 645.0721])

    class_info = {  # 标注信息
        "shaqiu": 1,
        "xintan": 2,
        "gengdi": 3,
        "building": 4,
        "water": 5,
        "veg": 6
    }
    n_class = len(class_info.items())

    def __init__(self, dataset_dir, alpha=0.3, random_seed=2024):
        self.dataset_dir = dataset_dir
        self.val_rate = alpha
        self.rand_seed = random_seed
        self.train_set, self.val_set = self._load_dataset()
        self.train_data_len = len(self.train_set)
        self.valid_data_len = len(self.val_set)

    def _load_dataset(self):
        fids = [f for f in os.listdir(os.path.join(self.dataset_dir, 'images')) if f[-4:] == ".tif"]
        class_sets = defaultdict(list)
        # 读取数据并按类别收集样本
        for f in fids:
            img, _, _ = read_gdal(os.path.join(self.dataset_dir, 'images', f))
            label, _, _ = read_gdal(os.path.join(self.dataset_dir, 'labels', f))
            assert img.shape[0:2] == label.shape[0:2], "样本数据中影像和标签栅格行列数必须一致"

            h, w, c = img.shape
            img = img.reshape((h * w, c))
            label = label.reshape((-1))
            for cls_name in self.class_info.keys():
                _indexes = (label == self.class_info[cls_name])
                _data = img[_indexes]
                class_sets[cls_name].extend(list(_data))

        if self.rand_seed is not None:
            random.seed(self.rand_seed)

        if True:
            negative_num = len(class_sets['target']) * 10
            _list = class_sets['others'].copy()
            class_sets['others'] = random.sample(_list, negative_num)

        # 每个类别按统一比例分为训练/验证样本数据
        train_set = defaultdict(list)
        val_set = defaultdict(list)
        print("{:<15} {:<10} {:<10}".format("class", "train", "val"))
        for cls_name, cls_id in self.class_info.items():
            _data = class_sets[cls_name]
            _total_num = len(_data)
            random.shuffle(_data)
            _train_num = int(_total_num * (1 - self.val_rate))
            _train_data = _data[:_train_num]
            _val_data = _data[_train_num:]
            train_set[cls_name].extend(_train_data)
            val_set[cls_name].extend(_val_data)
            print("{:<15} {:<10d} {:<10d}".format(cls_name, len(_train_data), len(_val_data)))
        return train_set, val_set

    @classmethod
    def Normalize(cls, img_data):
        return (img_data - cls.mean) / cls.std

    @classmethod
    def NDWI(cls, img_data):
        # img_data (N, 7)
        # NDWI: (G-NIR)/(G+NIR)
        g = img_data[..., 2].astype(np.float32)
        nir = img_data[..., 4].astype(np.float32)
        ndwi = np_divide((g - nir), (g + nir))
        return np.expand_dims(ndwi, axis=-1)

    @classmethod
    def NDVI(cls, img_data):
        # img_data (N, 7)
        # NDVI: (NIR-R)/(NIR+R)
        r = img_data[..., 3].astype(np.float32)
        nir = img_data[..., 4].astype(np.float32)
        ndvi = np_divide((nir - r), (nir + r))
        return np.expand_dims(ndvi, axis=-1)

    @classmethod
    def transform(cls, img_data, pca=None):
        # ndwi = cls.NDWI(img_data)
        ndvi = cls.NDVI(img_data)
        data = cls.Normalize(img_data)
        if pca is not None:
            data = pca.transform(data)
        data = np.concatenate([data, ndvi], axis=-1)
        return data

    def get_class_feature(self, class_name=''):
        if class_name in self.class_info:
            train_data = self.train_set[class_name]
            val_data = self.val_set[class_name]
        return np.array(train_data + val_data)

    def _get_samples(self, dataset, num_per_cls):
        image_data = []
        label_data = []
        if self.rand_seed is not None:
            random.seed(self.rand_seed)

        tmp = []
        for cls_name, cls_id in self.class_info.items():
            cls_data = dataset[cls_name]
            cls_label = list(np.ones(len(cls_data), np.uint8) * (cls_id - 1))
            cls_samples = list(zip(cls_label, cls_data))
            if num_per_cls is not None:
                _num = min(num_per_cls, len(cls_samples))
                cls_samples = random.sample(cls_samples, _num)
            tmp.extend(cls_samples)
        random.shuffle(tmp)

        for label_item, data_item in tmp:
            image_data.append(data_item)
            label_data.append(label_item)

        image_data, label_data = np.array(image_data), np.array(label_data)
        return image_data, label_data

    def get_train_set(self, num_per_cls=None):
        ''' data for trainning '''
        x_train, y_train = self._get_samples(self.train_set, num_per_cls)
        return x_train, y_train

    def get_val_set(self, num_per_cls=None):
        ''' data for valuation '''
        x_val, y_val = self._get_samples(self.val_set, num_per_cls)
        return x_val, y_val


class Sentinel2_Dataset(object):

    # 使用data_analysis/data_analysis.py的calculate_mean_std()计算得到
    mean = np.array([1059.0063, 1247.4547, 1220.6012, 1548.9696, 2078.7837, 2306.3306, 2250.63, 2383.973, 2008.6458, 1598.279])
    std = np.array([834.0967, 869.6389, 963.70483, 891.8735, 866.6485, 921.9942, 969.7178, 950.636, 917.763, 943.9702])

    class_info = {    # 标注信息
        "others": 1,
        "water": 2
    }
    n_class = len(class_info.items())

    def __init__(self, dataset_dir, alpha=0.5, random_seed=None):
        self.dataset_dir = dataset_dir
        self.val_rate = alpha
        self.rand_seed = random_seed
        self.train_set, self.val_set = self._load_dataset()
        self.train_data_len = len(self.train_set)
        self.valid_data_len = len(self.val_set)

    def _load_dataset(self):
        fids = [f for f in os.listdir(os.path.join(self.dataset_dir, 'images')) if f[-4:] == ".tif"]
        class_sets = defaultdict(list)
        # 读取数据并按类别收集样本
        for f in fids:
            img, _, _ = read_gdal(os.path.join(self.dataset_dir, 'images', f))
            label, _, _ = read_gdal(os.path.join(self.dataset_dir, 'labels', f))
            assert img.shape[0:2] == label.shape[0:2], "样本数据中影像和标签栅格行列数必须一致"

            h, w, c = img.shape
            img = img.reshape((h*w, c))
            label = label.reshape((-1))
            for cls_name in self.class_info.keys():
                _indexes = (label == self.class_info[cls_name])
                _data = img[_indexes]
                class_sets[cls_name].extend(list(_data))

        # 每个类别按统一比例分为训练/验证样本数据
        train_set = defaultdict(list)
        val_set = defaultdict(list)
        if self.rand_seed is not None:
            random.seed(self.rand_seed)
        print("{:<15} {:<10} {:<10}".format("class", "train", "val"))
        for cls_name, cls_id in self.class_info.items():
            _data = class_sets[cls_name]
            _total_num = len(_data)
            random.shuffle(_data)
            _train_num = int(_total_num * (1 - self.val_rate))
            _train_data = _data[:_train_num]
            _val_data = _data[_train_num:]
            train_set[cls_name].extend(_train_data)
            val_set[cls_name].extend(_val_data)
            print("{:<15} {:<10d} {:<10d}".format(cls_name, len(_train_data), len(_val_data)))
        return train_set, val_set

    @classmethod
    def Normalize(cls, img_data):
        return (img_data - cls.mean) / cls.std

    @classmethod
    def NDWI(cls, img_data):
        # img_data (N, 10)
        # NDWI: (G-NIR)/(G+NIR)
        G = img_data[:, 1].copy().astype(np.float32)
        NIR = img_data[:, 6].copy().astype(np.float32)
        index = (G - NIR) / (G + NIR)
        return index.reshape(-1, 1)

    @classmethod
    def NDVI(cls, img_data):
        # img_data (N, 10)
        # NDVI: (NIR-R)/(NIR+R)
        R = img_data[:, 2].copy().astype(np.float32)
        NIR = img_data[:, 6].copy().astype(np.float32)
        index = (NIR - R) / (NIR + R)
        return index.reshape(-1, 1)

    @classmethod
    def transform(cls, img_data, pca=None):
        ndwi = cls.NDWI(img_data)
        # ndvi = cls.NDVI(img_data)
        data = cls.Normalize(img_data)
        if pca is not None:
            data = pca.transform(data)
        data = np.concatenate([data, ndwi], axis=-1)
        return data

    def _get_samples(self, dataset, num_per_cls):
        image_data = []
        label_data = []
        if self.rand_seed is not None:
            self.rand_seed += 1
            random.seed(self.rand_seed)

        tmp = []
        for cls_name, cls_id in self.class_info.items():
            cls_data = dataset[cls_name]
            cls_label = list(np.ones(len(cls_data), np.uint8) * (cls_id - 1))
            cls_samples = list(zip(cls_label, cls_data))
            if num_per_cls is not None:
                _num = min(num_per_cls, len(cls_samples))
                cls_samples = random.sample(cls_samples, _num)
            tmp.extend(cls_samples)
        random.shuffle(tmp)

        for label_item, data_item in tmp:
            image_data.append(data_item)
            label_data.append(label_item)

        image_data, label_data = np.array(image_data), np.array(label_data)
        return image_data, label_data

    def get_train_set(self, num_per_cls=None):
        ''' data for trainning '''
        x_train, y_train = self._get_samples(self.train_set, num_per_cls)
        return x_train, y_train

    def get_val_set(self, num_per_cls=None):
        ''' data for valuation '''
        x_val, y_val = self._get_samples(self.val_set, num_per_cls)
        return x_val, y_val




#---------------------------------------------------
#   DataGenerator from txt
#---------------------------------------------------
class DataGeneratorFromTXT(object):
    '''
        数据生成器类
    '''
    def __init__(self, dataset_dir, class_num, alpha=0.1):
        self.dataset_dir = dataset_dir
        self.class_num = class_num
        n_class, self.cls_names, self.cls_colors, self.cls_samples, \
        self.train_set, self.val_set, self.train_num, self.val_num = self._load_dataset(dataset_dir, alpha)
        assert class_num == n_class


    def _load_dataset(self, dataset_dir, alpha):

        def get_information(info_str):
            assert info_str[1].split()[1] == 'Number'
            n_class = int(info_str[1].split()[-1])
            cls_names = []
            cls_colors = []
            cls_samples = []
            for i, l in enumerate(info_str):
                if 'name' in l:
                    # class name
                    cls_names.append(l.split()[-1].rstrip('\n'))
                    # class color
                    l2 = info_str[i + 1]
                    assert 'rgb' in l2
                    color = l2[17:]
                    color = color[1:-2]
                    cls_colors.append([int(v) for v in color.split(',')])
                    # class samples
                    l3 = info_str[i + 2]
                    assert 'npts' in l3
                    cls_samples.append(int(l3.split()[-1]))
            assert len(cls_names) == n_class
            return n_class, cls_names, cls_colors, cls_samples

        def get_class_data(data_str, indexs, cls_samples):
            class_datas = []
            for i, idx in enumerate(indexs):
                lines = data_str[idx: idx + cls_samples[i]]
                data = []
                for l in lines:
                    data.append([int(v) for v in l.split()[-32:]])
                class_datas.append(data)
            return class_datas

        print('\n> Data generator > load data ...')
        with open(dataset_dir, 'r') as f:
            lines = f.readlines()

        split = 0
        for l in lines:
            if l.startswith(';'):
                split += 1
            else:
                break
        info_str = lines[:split]
        data_str = lines[split:]

        n_class, cls_names, cls_colors, cls_samples = get_information(info_str)

        indexs = [0]
        indexs.extend([x + 1 for x in cls_samples[:-1]])
        indexs = np.cumsum(np.array(indexs))

        cls_datas = get_class_data(data_str, indexs, cls_samples)

        train_set = []
        val_set = []
        np.random.seed(10101)
        for i in range(n_class):
            _data = cls_datas[i]
            split = int(len(_data) * (1 - alpha))
            np.random.shuffle(_data)
            # (class_id, data)
            _train = [(i, d) for d in _data[:split]]
            train_set.extend(_train)
            _val = [(i, d) for d in _data[split:]]
            val_set.extend(_val)
        train_num = len(train_set)
        val_num = len(val_set)
        print('> Data generator > the number of train/val data: [ {} / {} ]\n'.format(train_num, val_num))
        return n_class, cls_names, cls_colors, cls_samples, train_set, val_set, train_num, val_num


    def gen_train(self):
        ''' data for trainning '''
        image_data = []
        label_data = []
        np.random.shuffle(self.train_set)
        for sample in self.train_set:
            label, data = sample
            image_data.append(data)
            label_data.append(label)
        # image_data = image_data[:5000]
        # label_data = label_data[:5000]
        return np.array(image_data), np.array(label_data)


    def gen_val(self):
        ''' data for validation '''
        image_data = []
        label_data = []
        np.random.shuffle(self.val_set)
        for sample in self.val_set:
            label, data = sample
            image_data.append(data)
            label_data.append(label)
        return np.array(image_data), np.array(label_data)


if __name__ == '__main__':
    pass
    generator = Sentinel2_Dataset(dataset_dir=r"D:\workspace\water\data\sentinel2\v1\trainval")

    train_data, train_label = generator.get_train_set(num_per_cls=1000)
    val_data, val_label = generator.get_val_set(num_per_cls=1000)

    print(train_data.shape, train_label.shape, val_data.shape, val_label.shape)
    for i in range(10):
        print(train_data[i], train_label[i])


    
