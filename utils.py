# -*- encoding: utf-8 -*-
# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------
import os
import numpy as np
from osgeo import gdal

# to solve the problem of 'ERROR 1: PROJ: pj_obj_create: Open of /opt/conda/share/proj failed'
os.environ['PROJ_LIB'] = r'C:\Users\HTHT\anaconda3\envs\torch17\Library\share\proj'


def np_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def mask_to_onehot(mask, n_class):
    """
    Converts a segmentation mask (H,W) to (H,W,K) where the last dim is a one
    hot encoding vector
    """
    H, W = mask.shape
    _onehot = np.eye(n_class)[mask.reshape(-1)]    # shape=(h*w, n_label),即长度为h*w的one-hot向量
    _onehot = _onehot.reshape(H, W, n_class)
    return _onehot


def onehot_to_mask(mask):
    """
    Converts a mask (H,W,K) to (H,W)
    """
    _mask = np.argmax(mask, axis=-1)
    # _mask[_mask != 0] += 1
    return _mask


def uint16_to_8(im_data, lower_percent=0.001, higher_percent=99.999):
    '''
        将uint 16bit转换成uint 8bit (压缩法)
    :param im_data: 图像矩阵(h, w, c)
    :type im_data: numpy
    :param lower_percent: np.percentile的最低百分位
    :type lower_percent: float
    :param higher_percent: np.percentile的最高百分位
    :type higher_percent: float
    :return: 返回图像矩阵(h, w, c)
    :rtype: numpy
    '''
    out = np.zeros_like(im_data, dtype=np.uint8)
    n = im_data.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(im_data[:, :, i], lower_percent)
        d = np.percentile(im_data[:, :, i], higher_percent)
        t = a + (im_data[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out


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


def write_gdal(im_data, path, im_proj=None, im_geotrans=None):
    '''
        重新写一个tiff图像
    :param im_data: 图像矩阵(h, w, c)
    :type im_data: numpy
    :param im_proj: 要设置的投影信息(默认None)
    :type im_proj: ?
    :param im_geotrans: 要设置的坐标信息(默认None)
    :type im_geotrans: ?
    :param path: 生成的图像路径(包括后缀名)
    :type path: string
    :return: None
    :rtype: None
    '''
    im_data = im_data.transpose((2, 0, 1))
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'float32' in im_data.dtype.name:
        datatype = gdal.GDT_Float32
    else:
        datatype = gdal.GDT_Float64
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        if im_geotrans == None or im_proj == None:
            pass
        else:
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


if __name__ == '__main__':
    pass
