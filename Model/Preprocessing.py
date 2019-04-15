from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import os


class Preprocessing(object):

    def __init__(self):
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()

    def caseFolding(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"""'""", '', sentence)
        sentence = re.sub(r'[^a-z]', ' ', sentence)
        return sentence

    def tokenisasi(self, sentence):
        return sentence.split()

    def stemming(self, token):
        return self.stemmer.stem(token)

    def stopwordRemoval(self, token):
        stopword = [line.rstrip('\n\r') for line in open(os.getcwd() + '/Model/stopwords.txt')]
        temp = []
        for i in range(len(token)):
            if token[i] not in stopword:
                temp.append(token[i])
        return temp
