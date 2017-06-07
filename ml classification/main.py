from python_speech_features import mfcc
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import scipy.io.wavfile as wav
import os
import numpy as np
from pydub import AudioSegment
import time
import ConfigParser
import re



def writeXYtrainset(path):
    X = []
    Y = ['b']*80 + ['c']*80 + ['p']*80 + ['r']*80
    print "Training."
    for filename in os.listdir(path):
        current_file = path + '%s' % filename
        print "working with " + current_file
        (rate,sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:25985]
        X.append(mfcc_feat[:])
    X = np.array(X)
    Y = np.array(Y)
    np.save('trainset.npy', X)
    np.save('trainclasses.npy', Y)
    return X, Y


def segment_song(path, timestart, timefinish):
    current_file = path.split('/')[-1]
    newAudio = AudioSegment.from_wav(path)
    seg = newAudio[int(timestart):int(timefinish)]
    seg.export('segof%s' % current_file, format="wav")

    (rate, sig) = wav.read('segof%s' %current_file)
    mfcc_feat = mfcc(sig, rate)
    mfcc_feat = np.reshape(mfcc_feat, -1)
    mfcc_feat = mfcc_feat[:25985]
    X = np.array(mfcc_feat)
    return X

def full_song(path):
    current_file = path
    (rate, sig) = wav.read(current_file)
    mfcc_feat = mfcc(sig,rate)
    mfcc_feat = np.reshape(mfcc_feat, -1)
    mfcc_feat = mfcc_feat[:25985]
    X = np.array(mfcc_feat)
    return X

def predict_naiveb(x, y, xx, path):
    model = GaussianNB()
    model.fit(x, y)
    print "Predicting with naive bayes."
    predicted = model.predict(xx)
    print path, 'predicted to be ', predicted

def predict_linear_SVM(x, y, xx, path):
    lin_clf = LinearSVC()
    lin_clf.fit(x, y)
    print "Predicting with linear SVM."
    predicted = lin_clf.predict(xx)
    print path, 'predicted to be ', predicted

def predict_neighbours(x, y, xx, path):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    print "Predicting with neighbours."
    predicted = neigh.predict(xx)
    print path, 'predicted to be ', predicted

def predict_RF(x, y, xx, path):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x, y)
    print "Predicting with random forest."
    predicted = clf.predict(xx)
    print path, 'predicted to be ', predicted

def predict_RC(x, y, xx, path):
    clf = RidgeClassifier()
    clf.fit(x, y)
    print "Predicting with ridge classifier."
    predicted = clf.predict(xx)
    print path, 'predicted to be ', predicted




def main():
    print "Welcome to music genre classification."
    conf = ConfigParser.RawConfigParser()
    conf.read("cnf.conf")

    #Trining phase
    path_train = conf.get('train', 'train_path')
    if conf.get('train', 'newtrainset') == 'False':
        print "Using default training set."
        trainfile = conf.get("train", "trainfile")
        trainclasses = conf.get("train", "trainclasses")
    else:
        writeXYtrainset(path_train)
        trainfile = conf.get("train", "trainfile")
        trainclasses = conf.get("train", "trainclasses")

    path_test = conf.get('testfile', 'test_path')
    folder = conf.get('segments', 'folder')

    x = np.load(trainfile)
    y = np.load(trainclasses)
    if folder == 'False':
        s = conf.get('segments', 'seg')
        alg = conf.get('testfile', 'alg')
        if s == 'True':
            tstart = conf.get('segments', 'timestart')
            tfinish = conf.get('segments', 'timefinish')
            xx = segment_song(path_test, tstart, tfinish)
        else:
            xx = full_song(path_test)
        if alg == 'nb':
            predict_naiveb(x, y, xx, path_test)
        elif alg == 'lsvm':
            predict_linear_SVM(x, y, xx, path_test)
        elif alg == 'knn':
            predict_neighbours(x, y, xx, path_test)
        elif alg == 'rf':
            predict_RF(x, y, xx, path_test)
        elif alg == 'rc':
            predict_RC(x, y, xx, path_test)
        elif alg == 'all':
            predict_naiveb(x, y, xx, path_test)
            predict_linear_SVM(x, y, xx, path_test)
            predict_neighbours(x, y, xx, path_test)
            predict_RF(x, y, xx, path_test)
            predict_RC(x, y, xx, path_test)
        else:
            print "Wrong input."
    else:
        alg = conf.get('testfolder', 'alg')
        s = conf.get('segments', 'seg')
        path_test = conf.get('testfolder', 'test_path')
        for filename in os.listdir(path_test):
            current_file = path_test + "%s" % filename
            if s == 'True':
                tstart = conf.get('segments', 'timestart')
                tfinish = conf.get('segments', 'timefinish')
                xx = segment_song(current_file, tstart, tfinish)
            else:
                xx = full_song(current_file)
            if alg == 'nb':
                predict_naiveb(x, y, xx, current_file)
            elif alg == 'lsvm':
                predict_linear_SVM(x, y, xx, current_file)
            elif alg == 'knn':
                predict_neighbours(x, y, xx, current_file)
            elif alg == 'rf':
                predict_RF(x, y, xx, current_file)
            elif alg == 'rc':
                predict_RC(x, y, xx, current_file)
            elif alg == 'all':
                predict_naiveb(x, y, xx, current_file)
                predict_linear_SVM(x, y, xx, current_file)
                predict_neighbours(x, y, xx, current_file)
                predict_RF(x, y, xx, current_file)
                predict_RC(x, y, xx, current_file)
            else:
                print "Wrong input."


main()
