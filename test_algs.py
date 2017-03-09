from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from pybrain.tools.shortcuts import buildNetwork
import scipy.io.wavfile as wav
import os
import numpy as np



def getXYtrainset():
    ''' getting features from train set'''
    path = "C:/Users/Lena/PycharmProjects/nir/train/"
    X = []
    Y = ['b']*90 + ['c']*90 + ['p']*90 + ['r']*90   # 4 classes: blues, classical, pop, rock

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate,sig) = wav.read(current_file)
        fbank_feat = logfbank(sig, rate)
        X.append(fbank_feat[1:3,:])

    X = np.array(X)
    dataset_size = len(X)
    newX = X.reshape(dataset_size, -1)
    return newX, Y


def predict_naiveb(x,y):
    ''' naive bayesian classifier'''
    model = GaussianNB()
    m = model.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig, rate)
        fbank_feat = logfbank(sig, rate)
        X = np.array(fbank_feat[1:3,:])
        newX = np.reshape(X, len(X[0])+len(X[1]))

        predicted = model.predict(newX)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_SVM(x,y):
    ''' Support Vector Machine classifier'''
    clf = SVC()
    clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        fbank_feat = logfbank(sig, rate)
        X = np.array(fbank_feat[1:3,:])
        newX = np.reshape(X, len(X[0])+len(X[1]))

        predicted = clf.predict(newX)
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
        fbank_feat = logfbank(sig, rate)
        X = np.array(fbank_feat[1:3,:])
        newX = np.reshape(X, len(X[0])+len(X[1]))

        predicted = clf.predict(newX)
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
        fbank_feat = logfbank(sig, rate)
        X = np.array(fbank_feat[1:3,:])
        newX = np.reshape(X, len(X[0])+len(X[1]))

        predicted = lin_clf.predict(newX)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_neighbours(x, y):
    '''k nearest neighbors classifier'''
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        fbank_feat = logfbank(sig, rate)
        X = np.array(fbank_feat[1:3,:])
        newX = np.reshape(X, len(X[0])+len(X[1]))

        predicted = neigh.predict(newX)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_RF(x, y):
    '''Random Forest'''
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        fbank_feat = logfbank(sig, rate)
        X = np.array(fbank_feat[1:3,:])
        newX = np.reshape(X, len(X[0])+len(X[1]))

        predicted = clf.predict(newX)
        print current_file, " predicted to be ", predicted
        if predicted[0] == filename[0]:
            i += 1
    print "total predicted correct: ", i

def predict_RC(x, y):
    '''ridge classifier'''
    clf = RidgeClassifier()
    clf.fit(x, y)
    i = 0
    path = "C:/Users/Lena/PycharmProjects/nir/test/"

    for filename in os.listdir(path):
        current_file = path + "%s" % filename
        print "working with " + current_file
        (rate, sig) = wav.read(current_file)
        fbank_feat = logfbank(sig, rate)
        X = np.array(fbank_feat[1:3,:])
        newX = np.reshape(X, len(X[0])+len(X[1]))

        predicted = clf.predict(newX)
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
