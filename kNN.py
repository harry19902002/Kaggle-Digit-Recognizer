
# coding: utf-8

# In[31]:

from numpy import *
import csv as csv
import numpy as np
import operator  
from os import listdir 


# ## kNN Algorithm

# In[32]:

def classify0 (inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX , (dataSetSize,1)).astype('float')  - dataSet.astype('float')
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


# ## Read Train data and Test data

# In[33]:

train_file = open('train.csv','rb')
train_file_object = csv.reader(train_file)
header_train = train_file_object.next()


# In[34]:

test_file = open('test.csv','rb')
test_file_object =csv.reader(test_file)
header_test = test_file_object.next()


# In[35]:

train_data = []
test_data = []
train_row_count = 0
test_row_count = 0


# In[36]:

for row in train_file_object:
    train_data.append(row)
for row in test_file_object:
    test_data.append(row)
train_data = np.array(train_data)
test_data = np.array(test_data)
train_row_count = sum(1 for row in train_data)
test_row_count = sum(1 for row in test_data)
print train_row_count
print test_row_count

# ## Training...

# In[39]:

hwLabels = []
trainingMat = zeros((42000,len(train_data[0])-1))


# In[40]:

for i in range(42000):
    classNumStr = train_data[i][0]
    hwLabels.append(classNumStr)
    trainingMat[i][:] = train_data[i][1::]
print "Training over"

# ## Generate the output csv

#trainingMat[trainingMat.astype(np.int) >= 150] = 1
#trainingMat[trainingMat.astype(np.int) < 150] = 0
#test_data[test_data.astype(np.int) >= 150] = 1
#test_data[test_data.astype(np.int) < 150] = 0
#print "Data preproccess over"
# In[ ]:

predictions_file = open('recognition.csv','wb')
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["ImageId","Label"])


# ## Testing
print "Begin Testing"
# In[ ]:

for i in range(test_row_count):
    classifierResult = classify0(test_data[i],trainingMat,hwLabels,3)
    predictions_file_object.writerow([i+1,int(classifierResult)])
    print "Proccess finish %d//28000 " %i
print "All finished !!!"