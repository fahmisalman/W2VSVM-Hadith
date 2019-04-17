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

    # Train phase
    Data_Proc.data_process()

    x_train = np.array(Word2Vec.fit(import_csv(os.getcwd() + '/Model/Data/Data_train_preprocessing_k-5.csv')))
    y_train = np.array(import_csv(os.getcwd() + '/Model/Data/Target_train_k-5.csv'))

    y_train = np.int_(y_train)

    kernel_svm = ['linear', 'poly', 'rbf', 'sigmoid']

    for ii in range(len(kernel_svm)):

        for i in range(3):
            model = SVM.fit(x_train, y_train[:, i], kernel=kernel_svm[ii])
            joblib.dump(model, os.getcwd() + '/Model/SVM_Model/Model_{}.pkl'.format(i + 1))

        # Test phase

        x = import_csv(os.getcwd() + '/Model/Data/Data_test_k-5.csv')[0]
        for i in range(len(x)):
            x[i] = Data_Proc.preprocessing(x[i])

        x_test = np.array(Word2Vec.fit(x))
        y_test = np.array(import_csv(os.getcwd() + '/Model/Data/Target_test_k-5.csv'))

        y_test = np.int_(y_test)

        score = 0

        for i in range(3):
            model = joblib.load(os.getcwd() + '/Model/SVM_Model/Model_{}.pkl'.format(i + 1))
            score += model.score(x_test, y_test[:, i])

        print('Akurasi\t\t\t: {}'.format(score/3))
        print('Hamming loss\t: {}'.format(1-(score/3)))
