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




def getXYtrainset(path):
    X = []
    Y = ['b']*80 + ['c']*80 + ['p']*80 + ['r']*80
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


def predict_naiveb(x, y, path, folder=False):
    model = GaussianNB()
    model.fit(x, y)
    if folder == False:
        current_file = path
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = model.predict(X)
        print current_file, " predicted to be ", predicted
    else:
        for filename in os.listdir(path):
            current_file = path + "%s" % filename
            (rate, sig) = wav.read(current_file)
            mfcc_feat = mfcc(sig,rate)
            mfcc_feat = np.reshape(mfcc_feat, -1)
            mfcc_feat = mfcc_feat[:77974]
            X = np.array(mfcc_feat)
            predicted = model.predict(X)
            print current_file, " predicted to be ", predicted

def predict_SVM(x, y, path, folder=False):
    clf = SVC()
    clf.fit(x, y)
    if folder == False:
        current_file = path
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = clf.predict(X)
        print current_file, " predicted to be ", predicted
    else:
        for filename in os.listdir(path):
            current_file = path + "%s" % filename
            (rate, sig) = wav.read(current_file)
            mfcc_feat = mfcc(sig,rate)
            mfcc_feat = np.reshape(mfcc_feat, -1)
            mfcc_feat = mfcc_feat[:77974]
            X = np.array(mfcc_feat)
            predicted = clf.predict(X)
            print current_file, " predicted to be ", predicted

def predict_linear_SVM(x, y, path, folder=False):
    lin_clf = LinearSVC()
    lin_clf.fit(x, y)
    if folder == False:
        current_file = path
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = lin_clf.predict(X)
        print current_file, " predicted to be ", predicted
    else:
        for filename in os.listdir(path):
            current_file = path + "%s" % filename
            (rate, sig) = wav.read(current_file)
            mfcc_feat = mfcc(sig,rate)
            mfcc_feat = np.reshape(mfcc_feat, -1)
            mfcc_feat = mfcc_feat[:77974]
            X = np.array(mfcc_feat)
            predicted = lin_clf.predict(X)
            print current_file, " predicted to be ", predicted

def predict_neighbours(x, y, path, folder=False):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    if folder == False:
        current_file = path
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = neigh.predict(X)
        print current_file, " predicted to be ", predicted
    else:
        for filename in os.listdir(path):
            current_file = path + "%s" % filename
            (rate, sig) = wav.read(current_file)
            mfcc_feat = mfcc(sig,rate)
            mfcc_feat = np.reshape(mfcc_feat, -1)
            mfcc_feat = mfcc_feat[:77974]
            X = np.array(mfcc_feat)
            predicted = neigh.predict(X)
            print current_file, " predicted to be ", predicted

def predict_RF(x, y, path, folder=False):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x, y)
    if folder == False:
        current_file = path
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = clf.predict(X)
        print current_file, " predicted to be ", predicted
    else:
        for filename in os.listdir(path):
            current_file = path + "%s" % filename
            (rate, sig) = wav.read(current_file)
            mfcc_feat = mfcc(sig,rate)
            mfcc_feat = np.reshape(mfcc_feat, -1)
            mfcc_feat = mfcc_feat[:77974]
            X = np.array(mfcc_feat)
            predicted = clf.predict(X)
            print current_file, " predicted to be ", predicted

def predict_RC(x, y, path, folder=False):
    clf = RidgeClassifier()
    clf.fit(x, y)
    if folder == False:
        current_file = path
        (rate, sig) = wav.read(current_file)
        mfcc_feat = mfcc(sig,rate)
        mfcc_feat = np.reshape(mfcc_feat, -1)
        mfcc_feat = mfcc_feat[:77974]
        X = np.array(mfcc_feat)
        predicted = clf.predict(X)
        print current_file, " predicted to be ", predicted
    else:
        for filename in os.listdir(path):
            current_file = path + "%s" % filename
            (rate, sig) = wav.read(current_file)
            mfcc_feat = mfcc(sig,rate)
            mfcc_feat = np.reshape(mfcc_feat, -1)
            mfcc_feat = mfcc_feat[:77974]
            X = np.array(mfcc_feat)
            predicted = clf.predict(X)
            print current_file, " predicted to be ", predicted




def main():
    train = str(raw_input("Welcome to music genre classification. If you want to change training set type 'yes' (or type 'no' if you don't): "))
    if train == 'no':
        path_train = "C:/Users/Lena/PycharmProjects/nir/train/"
        answer = raw_input("Type 'file' or 'folder' if you want to classify one or multiple files. ")
        if answer == 'file':
            path = raw_input("Type path to the file: ")
            print "Supported ml algorithms: naive bayesian, SVM, linear SVM, neighbours, random forest and ridge classifier."
            alg = raw_input("Choose algorithm for your classification (nb, svm, lsvm, knn, rf, rc or all): ")
            if alg == 'nb':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with naive bayesian."
                predict_naiveb(x, y, path)
            elif alg == 'svm':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with SVM."
                predict_SVM(x, y, path)
            elif alg == 'lsvm':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with linear SVM."
                predict_linear_SVM(x, y, path)
            elif alg == 'knn':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with neighbours."
                predict_neighbours(x, y, path)
            elif alg == 'rf':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with random forest."
                predict_RF(x, y, path)
            elif alg == 'rc':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with ridge classifier."
                predict_RC(x, y, path)
            elif alg == 'all':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with naive bayesian."
                predict_naiveb(x, y, path)
                print "Predicting with SVM."
                predict_SVM(x, y, path)
                print "Predicting with linear SVM."
                predict_linear_SVM(x, y, path)
                print "Predicting with neighbours."
                predict_neighbours(x, y, path)
                print "Predicting with random forest."
                predict_RF(x, y, path)
                print "Predicting with ridge classifier."
                predict_RC(x, y, path)

        elif answer == 'folder':
            path = raw_input("Type path to the folder: ")
            print "Supported ml algorithms: naive bayesian, SVM, linear SVM, neighbours, random forest and ridge classifier."
            alg = raw_input("Choose algorithm for your classification (nb, svm, lsvm, knn, rf, rc or all): ")
            if alg == 'nb':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with naive bayesian."
                predict_naiveb(x, y, path, folder=True)
            elif alg == 'svm':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with SVM."
                predict_SVM(x, y, path, folder=True)
            elif alg == 'lsvm':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with linear SVM."
                predict_linear_SVM(x, y, path, folder=True)
            elif alg == 'knn':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with neighbours."
                predict_neighbours(x, y, path, folder=True)
            elif alg == 'rf':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with random forest."
                predict_RF(x, y, path, folder=True)
            elif alg == 'rc':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with ridge classifier."
                predict_RC(x, y, path, folder=True)
            elif alg == 'all':
                print "Training."
                x, y = getXYtrainset(path_train)
                print "Predicting with naive bayesian."
                predict_naiveb(x, y, path, folder=True)
                print "Predicting with SVM."
                predict_SVM(x, y, path, folder=True)
                print "Predicting with linear SVM."
                predict_linear_SVM(x, y, path, folder=True)
                print "Predicting with neighbours."
                predict_neighbours(x, y, path, folder=True)
                print "Predicting with random forest."
                predict_RF(x, y, path, folder=True)
                print "Predicting with ridge classifier."
                predict_RC(x, y, path, folder=True)
        else:
            print "Wrong input."
    else:
        continue
        '''rain_path = raw_input("Plese type the path to the train set: ")
        getXYtrainset_new(train_path)
        #ask for classes
        #return song-class info
        '''

main()
