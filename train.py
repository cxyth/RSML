# -*- encoding: utf-8 -*-
# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------

import os
import sys
import time
import numpy as np
import os.path as osp
from operator import truediv
from tqdm import tqdm
from models import SVM_RBF, DecisionTree, RandomForest
from datasets import Landsat_Dataset
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.externals import joblib  # scikit-learn <= 0.21
import joblib   # scikit-learn > 0.21


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, path):
    f = open(path, 'w')
    lines = []
    lines.append('OAs for each iteration are:' + str(oa_ae) + '\n')
    lines.append('AAs for each iteration are:' + str(aa_ae) + '\n')
    lines.append('KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n')
    lines.append('mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n')
    lines.append('mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n')
    lines.append('mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n')
    lines.append('Total average Training time is: ' + str(np.sum(training_time_ae)) + '\n')
    lines.append('Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n')
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    lines.append("Mean of all elements in confusion matrix: " + str(element_mean) + '\n')
    lines.append("Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n')
    [print(l) for l in lines]
    f.writelines(lines)
    f.close()

if __name__ == '__main__':
    
    dataset_dir = "../datasets/v1"
    pca_model = "./runs/pca.v1/pca.model"
    logdir = osp.join(os.getcwd(), 'runs', 'rf.v1.pca_ndvi')

    os.makedirs(logdir, exist_ok=True)

    dataset = Landsat_Dataset(dataset_dir=dataset_dir, alpha=0.3, random_seed=2024)
    n_class = dataset.n_class

    # PCA模型
    pca = joblib.load(pca_model)

    print('-----Importing Setting Parameters-----')

    ITER = 3    # 训练次数，可多次训练查看平均性能
    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((ITER, n_class))

    for index_iter in tqdm(range(ITER)):
        # train
        x_train, y_train = dataset.get_train_set(num_per_cls=400)
        x_train = dataset.transform(x_train, pca)
        x_val, y_val = dataset.get_val_set(num_per_cls=400)
        x_val = dataset.transform(x_val, pca)
        # net = SVM_RBF(x_train, y_train)
        net = RandomForest(x_train, y_train)
        t0 = time.time()
        model = net.train()
        t1 = time.time()
        # joblib.dump(model, osp.join(logdir, 'rf_{:02d}.model'.format(index_iter)))
        joblib.dump(model, osp.join(logdir, 'm{:02d}.model'.format(index_iter)))
        # eval
        preds = []
        for i in range(x_val.shape[0]):
            _pred = model.predict(x_val[i].reshape((1, -1)))
            preds.extend(_pred)
        t2 = time.time()

        # collections.Counter(preds)
        overall_acc = metrics.accuracy_score(preds, y_val)
        confusion_matrix = metrics.confusion_matrix(preds, y_val)
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(preds, y_val)

        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TRAINING_TIME.append(t1 - t0)
        TESTING_TIME.append(t2 - t1)
        ELEMENT_ACC[index_iter, :] = each_acc
    print("--------" + net.name + " Training Finished-----------")
    _path = osp.join(logdir, 'eval.txt')
    record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, _path)

