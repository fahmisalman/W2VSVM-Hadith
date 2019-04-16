from gensim.models import Word2Vec
import numpy as np


def get_w2v_model(size, window):
    return Word2Vec.load('Model/W2V_Model/sg_hadith_size={}_window={}.model'.format(size, window))


def to_vector(data, model, size):
    vector = np.zeros(size)
    for word in data:
        if word in model.wv.vocab:
            vector += model[word]
    return vector


def fit(data, size=300, window=5):

    vectors = []
    model = get_w2v_model(size, window)
    for row in data:
        vectors.append(to_vector(row, model, size))
    return vectors
