from sklearn.svm import SVC


def fit(x, y, kernel='rbf'):

    clf = SVC(kernel=kernel)
    return clf.fit(x, y)


def score(clf, x, y):
    return clf.score(x, y)
