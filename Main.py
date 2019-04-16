from Model import Data_Proc
from Model import Word2Vec
from Model import SVM
from sklearn.externals import joblib
import numpy as np
import csv
import os


def import_csv(loc):
    x = []
    with open(loc) as f:
        file = csv.reader(f)
        for row in file:
            x.append(row)
    return x


if __name__ == '__main__':

    Data_Proc.data_process()

    x_train = np.array(Word2Vec.fit(import_csv(os.getcwd() + '/Model/Data/Data_train_preprocessing_k-5.csv')))
    y_train = np.array(import_csv(os.getcwd() + '/Model/Data/Target_train_k-5.csv'))

    y_train = np.int_(y_train)

    for i in range(3):
        model = SVM.fit(x_train, y_train[:, i])
        joblib.dump(model, os.getcwd() + '/Model/SVM_Model/Model_{}.pkl'.format(i + 1))

