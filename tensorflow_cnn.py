import numpy as np
import csv as csv
import tensorflow as tf

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from data_process import loadTrainData
from data_process import loadTestData
from data_process import writeCSV

# Constants
NUM_IMAGE_PIXELS = 784
IMAGE_WIDTH = 28
NUM_DIGITS = 10

count = 0

#Load Data
trainNumber = 42000
testNumber = 28000
trainData,trainLabel = loadTrainData(trainNumber)
testData = loadTestData(testNumber)
#test_results = None

print "Load data over"

def train_laber_convert(trainLabel):
	label = list()
	for row in trainLabel:
		values = np.zeros(10)
		values[row.astype(int)] = 1
		label.append(values)
	newLabel = np.array(label, dtype='float32')
	return newLabel

def get_train_batch(trainLabel ,trainData, batch_size):
	global count

	if batch_size > trainNumber:
		return None, None
	elif (trainNumber - count) > batch_size:
		batch_features = trainData[count:(count + batch_size),]
		batch_labels = trainLabel[count:(count + batch_size),]
		count = count + batch_size
		#print(batch_labels[15])
		#print batch_labels
		return batch_features, batch_labels
	else:
		nbr_gap = batch_size - (trainNumber - count)
		batch_features = np.vstack((trainData[count:trainNumber, ],
			trainData[0:nbr_gap, ]))
		batch_labels = np.vstack((trainLabel[count:trainNumber, ],
			trainLabel[0:nbr_gap, ]))
		count = 0
		return batch_features, batch_labels

def get_test_batch(testData, batch_size):
	global count
	if batch_size > testNumber:
		return None
	elif (testNumber - count) >= batch_size:
		batch_data = testData[count:(count + batch_size)]
		count = count + batch_size
		return batch_data
	else:
		return None

#Weight Initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	inital = tf.constant(0.1,shape = shape)
	return tf.Variable(inital)

#Convolution and Pooling
def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')

newtrainLabel = train_laber_convert(trainLabel)

#create a Session
sess = tf.InteractiveSession()

# Declare placeholders for the x data batches and the y_ correct answer batches
x = tf.placeholder(tf.float32, shape=[None, NUM_IMAGE_PIXELS])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_DIGITS])

#First Convolutional Layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = weight_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])
#convolve x_image with weight tensor
#apply ReLU ,max_pool function
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Train and Evaluate the Model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(30000):
	batch_xs, batch_ys = get_train_batch(newtrainLabel,trainData,50)
	if i%500 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		print("step %d, training accuracy %f"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

#Use the Train result to predict
result = list()
test_batch_size = 100
count = 0

for j in xrange(testNumber / test_batch_size):
	batch_data = get_test_batch(testData,100)
	if batch_data is not None:
		y_predict = tf.argmax(y_conv, 1)
		test_results = sess.run(y_predict, feed_dict={x: batch_data, keep_prob: 1.0})
		for i in range(test_batch_size):
			result.append(test_results[i])
		print("Test step %d is over." %j)

sess.close()
writeCSV('tensorflow_prediction.csv',result,testNumber)
#print np.shape(result)