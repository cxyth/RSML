# Remote Sensing with Machine Learning
  
## 环境：
- scikit-learn
- joblib
- gdal

## 运行：
#### 训练
```shell
    python train.py
```

#### 预测
```shell
    python predict.py
```

## 用于自己的数据：
当使用此工程用于自己的数据时，通常步骤如下：  
1、 准备好训练用的数据集，例如如果是栅格数据的话可将数据集整理为以下目录结构：
```shell
    ├── mydataset
        ├── images                 # 存放原始影像
            ├── xxx01.tif
            ├── xxx02.tif
        ├── labels               # 存放同名的栅格化的标签文件
            ├── xxx01.tif
            ├── xxx02.tif
```

2、在datasets.py中实现加载数据的方法，可参考'Dataset'类的定义，其中样本的均值和方差可以使用calculate_mean_std()进行统计。 
   
3、修改 train.py：
```shell
    ...
    from datasets import Dataset                                # 在头部引用自定义的数据加载类
    ...
    ...
    dataset_dir = r"D:\project\mydataset"      # 指定数据集路径
    logdir = osp.join(os.getcwd(), 'runs', 'svm.v2.1')          # 指定模型保存路径，可将'svm.v2.1'改为自定义的文件夹
```
  
4、修改 predict.py：
```shell
    ...
    MODEL_PATH = r"D:\project\SVM\runs\svm.v1.1\svm_00.model"       # 引用的模型路径
    DATA_PATH = r"D:\project\test_data"                                  # 输入数据路径
    OUTPUT_PATH = r"D:\project\SVM\runs\svm.v1.1\svm_00_pred"       # 结果保存路径
    ...
    ...
    TEST_SET = sorted([os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.tif')])    # 测试数据的读取方式看情况调整
    ...
```