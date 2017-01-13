import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bn_class import *


# Hyperparameters
num_filt_1 = 15  # Number of filters in first conv layer
num_filt_2 = 8  # Number of filters in second conv layer
num_filt_3 = 8  # Number of filters in thirs conv layer
num_fc_1 = 40  # Number of neurons in hully connected layer
max_iterations = 20000
batch_size = 100
dropout = 0.5  # Dropout rate in the fully connected layer
plot_row = 5  # How many rows do you want to plot in the visualization
regularization = 1e-4
learning_rate = 2e-3
input_norm = False  # Do you want z-score input normalization?


# load data
dataset = "2330"
datadir = 'data/'+ dataset
data_train = np.loadtxt(datadir+'_train_ma',delimiter=',')
data_test_val = np.loadtxt(datadir+'_test_ma',delimiter=',')


# split training and testing data
X_train = data_train[:,1:]
X_test = data_test_val[:,1:]
N = X_train.shape[0]
Ntest = X_test.shape[0]
D = X_train.shape[1]
y_train = data_train[:,0]
y_test = data_test_val[:,0]


# normalize x and y
num_classes = len(np.unique(y_train))
base = np.min(y_train)  #Check if data is 0-based
if base != 0:
    y_train -=base
    y_test -= base

if input_norm:
    mean = np.mean(X_train,axis=0)
    variance = np.var(X_train,axis=0)
    X_train -= mean
    #The 1e-9 avoids dividing by zero
    X_train /= np.sqrt(variance)+1e-9
    X_test -= mean
    X_test /= np.sqrt(variance)+1e-9

epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))


# place for the input variables
x = tf.placeholder("float", shape=[None, D], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
keep_prob = tf.placeholder("float")
bn_train = tf.placeholder(tf.bool)  #Boolean value to guide batchnorm


# w and b and conv function
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("Reshaping_data") as scope:
    x_image = tf.reshape(x, [-1,D,1,1])


# Build the graph
# ewma is the decay for which we update the moving average of the 
# mean and variance in the batch-norm layers
with tf.name_scope("Conv1") as scope:
    W_conv1 = weight_variable([4, 1, 1, num_filt_1], 'Conv_Layer_1')
    b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
    a_conv1 = conv2d(x_image, W_conv1) + b_conv1
  
with tf.name_scope('Batch_norm_conv1') as scope:
    ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
    bn_conv1 = ConvolutionalBatchNormalizer(num_filt_1, 0.001, ewma, True)           
    update_assignments = bn_conv1.get_assigner() 
    a_conv1 = bn_conv1.normalize(a_conv1, train=bn_train) 
    h_conv1 = tf.nn.relu(a_conv1)
  
with tf.name_scope("Conv2") as scope:
    W_conv2 = weight_variable([4, 1, num_filt_1, num_filt_2], 'Conv_Layer_2')
    b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
    a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
  
with tf.name_scope('Batch_norm_conv2') as scope:
    bn_conv2 = ConvolutionalBatchNormalizer(num_filt_2, 0.001, ewma, True)           
    update_assignments = bn_conv2.get_assigner() 
    a_conv2 = bn_conv2.normalize(a_conv2, train=bn_train) 
    h_conv2 = tf.nn.relu(a_conv2)
    
with tf.name_scope("Conv3") as scope:
    W_conv3 = weight_variable([4, 1, num_filt_2, num_filt_3], 'Conv_Layer_3')
    b_conv3 = bias_variable([num_filt_3], 'bias_for_Conv_Layer_3')
    a_conv3 = conv2d(h_conv2, W_conv3) + b_conv3
  
with tf.name_scope('Batch_norm_conv3') as scope:
    bn_conv3 = ConvolutionalBatchNormalizer(num_filt_3, 0.001, ewma, True)           
    update_assignments = bn_conv3.get_assigner() 
    a_conv3 = bn_conv3.normalize(a_conv3, train=bn_train) 
    h_conv3 = tf.nn.relu(a_conv3)

with tf.name_scope("Fully_Connected1") as scope:
    W_fc1 = weight_variable([D*num_filt_3, num_fc_1], 'Fully_Connected_layer_1')
    b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
    h_conv3_flat = tf.reshape(h_conv3, [-1, D*num_filt_3])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
  
with tf.name_scope("Fully_Connected2") as scope:
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = tf.Variable(tf.truncated_normal([num_fc_1, num_classes], stddev=0.1),name = 'W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  
 
with tf.name_scope("SoftMax") as scope:
    regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
                  tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) + 
                  tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + 
                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc2,y_)
    cost = tf.reduce_sum(loss) / batch_size
    cost += regularization*regularizers


# define train optimizer
with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)

    numel = tf.constant([[0]])
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient
      
      numel +=tf.reduce_sum(tf.size(variable))  
with tf.name_scope("Evaluating_accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(h_fc2,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# run session and evaluate performance
perf_collect = np.zeros((3,int(np.floor(max_iterations /100))))
merged = tf.merge_all_summaries()
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("/home/carine/Desktop/eventlog/", sess.graph_def)
    sess.run(tf.initialize_all_variables())
  
    step = 0  # Step is a counter for filling the numpy array perf_collect
    for i in range(max_iterations):#training process
		batch_ind = np.random.choice(N,batch_size,replace=False)
		
		if i==0:
		    acc_test_before = sess.run(accuracy, feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
		if i%100 == 0:
			#Check training performance
		    result = sess.run(accuracy,feed_dict = { x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False})
		    perf_collect[1,step] = result 
		    
		    #Check validation performance
		    result = sess.run(accuracy, feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
		    acc = result
		    perf_collect[0,step] = acc    
		    print(" Training accuracy at %s out of %s is %s" % (i,max_iterations, acc))
		    step +=1
		sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train : True})
    
          #training process done!
	result = sess.run([accuracy,numel], feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
  
    predict=sess.run(tf.argmax(h_fc2,1), feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
    pred_summ=tf.scalar_summary("prediction", predict)
    print("pred "+"real")  
    for x in xrange(0,len(predict)):
    	print(str(predict[x]+1)+"    "+str(int(y_test[x]+1)))
    acc_test = result[0]
    print('The network has %s trainable parameters'%(result[1]))

    writer.flush()

# show the graph of validation accuracy
# 2 for drop or same, 1 for rise 
print('The accuracy on the test data is %.3f, before training was %.3f' %(acc_test,acc_test_before))
plt.figure()
plt.plot(perf_collect[0],label='Valid accuracy')
plt.plot(perf_collect[1],label = 'Train accuracy')
plt.axis([0, step, 0, np.max(perf_collect)])
plt.show()
plt.figure()

