from sklearn.svm import SVC


def fit(x, y, kernel='rbf'):

    clf = SVC(kernel=kernel)
    clf.fit(x, y)
