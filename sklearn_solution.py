from numpy import *
import numpy as np
import csv as csv
import data_process

from data_process import loadTrainData
from data_process import loadTestData
from data_process import writeCSV
#use SVM
from sklearn import svm
def svcClassifiy(trainData,trainLabel,testData):
    svcClf = svm.SVC(kernel = 'linear')
    svcClf.fit(trainData,ravel(trainLabel))
    testLabel = svcClf.predict(testData)
    return testLabel

from sklearn.neighbors import KNeighborsClassifier  
def knnClassify(trainData,trainLabel,testData): 
    knnClf=KNeighborsClassifier() #default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData,ravel(trainLabel))
    testLabel=knnClf.predict(testData)
    return testLabel

from sklearn.naive_bayes import GaussianNB      #nb for Gaussian
def GaussianNBClassify(trainData,trainLabel,testData): 
    nbClf=GaussianNB()          
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    return testLabel

trainNumber = 42000
testNumber = 28000
trainData,trainLabel = loadTrainData(trainNumber)
testData = loadTestData(testNumber)

result = knnClassify(trainData,trainLabel,testData) 
#Write data
writeCSV('svm_prediction.csv',result,testNumber)