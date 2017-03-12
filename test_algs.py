from python_speech_features import mfcc
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import scipy.io.wavfile as wav
import os
import numpy as np
import time


def getXYtrainset():
    path = "C:/Users/Lena/PycharmProjects/nir/train/"
    X = []
    Y = ['b']*90 + ['c']*90 + ['p']*90 + ['r']*90

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate,sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X.append(mfcc_feat[:])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def predict_naiveb(x,y):
    model = GaussianNB()
    model.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = model.predict(X)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_SVM(x,y):
    clf = SVC()
    clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = clf.predict(X)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_NU_SVM(x, y):
    clf = NuSVC()
    clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = clf.predict(X)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_linear_SVM(x, y):
    lin_clf = LinearSVC()
    lin_clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = lin_clf.predict(X)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_neighbours(x, y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = neigh.predict(X)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_RF(x, y):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = clf.predict(X)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_RC(x, y):
    clf = RidgeClassifier()
    clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = clf.predict(X)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i



x, y = getXYtrainset()

predict_naiveb(x, y)      #total predicted correct:  19
predict_SVM(x, y)         #total predicted correct:  23
predict_NU_SVM(x, y)      #total predicted correct:  24
predict_linear_SVM(x, y)  #total predicted correct:  19
predict_neighbours(x, y)  #total predicted correct:  23(1), 23(2), 26 (3), 24(4)
predict_RF(x, y)          #total predicted correct:  21
predict_RC(x, y)          #total predicted correct:  27
