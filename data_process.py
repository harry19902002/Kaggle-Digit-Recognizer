import csv as csv
import numpy as np
from sklearn import preprocessing

def toInt(array):  
	array=np.mat(array)  
	m,n=np.shape(array)  
	newArray=np.zeros((m,n))  
	for i in xrange(m):  
		for j in xrange(n):  
			newArray[i,j]=int(array[i,j])  
	return newArray 

def toFloat(array):
	array=np.mat(array)  
	m,n=np.shape(array)
	newArray=np.zeros((m,n))  
	for i in xrange(m):  
		for j in xrange(n):  
			newArray[i,j]=float(array[i,j])
	return newArray

def toFloat255(array):
	array=np.mat(array)  
	m,n=np.shape(array)
	newArray=np.zeros((m,n))  
	for i in xrange(m):  
		for j in xrange(n):  
			newArray[i,j]=float(array[i,j])/255
	return newArray 

def loadTrainData(dataNumber):
	train_file = open('train.csv','rb')
	train_file_object = csv.reader(train_file)
	header_train = train_file_object.next()
	train_data = []
	for row in train_file_object:
		train_data.append(row)
	train_data = np.array(train_data)

	label = []
	dataMat = np.zeros((dataNumber,len(train_data[0])-1))

	for i in range(dataNumber):
		classNumStr = train_data[i][0]
		label.append(classNumStr)
		dataMat[i][:] = train_data[i][1::]

	dataMat = toFloat255(dataMat)
	normalized_dataMat = preprocessing.normalize(dataMat)

	return dataMat,label

def loadTestData(dataNumber):
	test_file = open('test.csv','rb')
	test_file_object = csv.reader(test_file)
	header_train = test_file_object.next()
	test_data = []
	count = 0
	for row in test_file_object:
		int_row = [float(i) for i in row]
		test_data.append(int_row)
		count += 1
		if(count > dataNumber):
			test_data = np.array(test_data)
			test_data = toFloat(test_data)
			normalized_test_data = preprocessing.normalize(test_data)
			return toFloat255(test_data)
	return test_data


def writeCSV(fileName,result,test_number):
	write_file = open(fileName,'wb')
	write_file_object = csv.writer(write_file)
	write_file_object.writerow(["ImageId","Label"])

	for i in range(test_number):
		write_file_object.writerow([i+1,int(result[i])])
