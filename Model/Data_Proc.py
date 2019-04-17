from Model import Preprocessing
from sklearn.model_selection import KFold
import numpy as np
import xlrd
import csv
import os


def load_data():
    workbook = xlrd.open_workbook(os.getcwd() + '/Dataset/Data Hadis.xlsx')
    worksheet1 = workbook.sheet_by_index(0)
    x = []
    y = []
    for i in range(2, 1068):
        x.append(worksheet1.cell(i, 2).value)
        y.append([int(worksheet1.cell(i, 4).value),
                  int(worksheet1.cell(i, 5).value),
                  int(worksheet1.cell(i, 6).value)])

    return x, y


def preprocessing(sentence):
    pre = Preprocessing.Preprocessing()
    sentence = pre.caseFolding(sentence)
    token = pre.tokenisasi(sentence)
    token = pre.stopwordRemoval(token)
    for i in range(len(token)):
        token[i] = pre.stemming(token[i])
    return token


def bag_of_words(data, label):

    temp = []
    bag = []
    for i in range(len(data)):
        temp.append(data[i])

    for i in range(len(temp)):
        for j in range(len(temp[i])):
            bag.append(temp[i][j])

    bag = list(set(bag))

    term = []
    target = []
    term.append(bag)
    for i in range(len(data)):
        temp = []
        for j in range(len(bag)):
            temp.append(data[i].count(bag[j]))
        target.append(label[i])
        term.append(temp)

    return term, target


def save_csv(loc, d):
    with open(loc, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(d)


def save_text_csv(loc, d):
    with open(loc, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows([d])


def data_process():

    X, y = load_data()
    X = np.array(X)
    y = np.array(y)

    k, i = 5, 1
    k_fold = KFold(n_splits=k)

    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        save_text_csv(os.getcwd() + '/Model/Data/Data_train_k-{}.csv'.format(i), X_train)
        save_text_csv(os.getcwd() + '/Model/Data/Data_test_k-{}.csv'.format(i), X_test)
        save_csv(os.getcwd() + '/Model/Data/Target_test_k-{}.csv'.format(i), y_test)

        x_train = []
        for j in range(len(X_train)):
            x_train.append(preprocessing(X_train[j]))

        save_csv(os.getcwd() + '/Model/Data/Data_train_preprocessing_k-{}.csv'.format(i), x_train)
        save_csv(os.getcwd() + '/Model/Data/Target_train_k-{}.csv'.format(i), y_train)

        i += 1
