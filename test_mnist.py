import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data',one_hot='True')
#build model
x=tf.placeholder('float',[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
log_y=tf.log(y)
#train model
y_=tf.placeholder('float',[None,10])
a=y_*log_y
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
#use gradientdescentOptimizer
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#initialization parameter
init=tf.initialize_all_variables()
#begin session valiable
sess=tf.Session()
sess.run(init)
#begin train
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#judge prediction whether correction
#tf.argmax() can give which tensor object in dimensional max index value
#tf.argmax(y,1)return that model for random input prediction label value
#tf.argmax(y_1) representative correct label
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#tf.cast will change boolean to float
accuracy =tf.reduce_mean(tf.cast(correct_prediction,'float'))
#calculation model in train_dataset correct rate
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
#build more layer convolution
#definition two function for initialization
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#convolution and pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pooling_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#implement first layer
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
#change x to four-dimensional vector for use
x_image=tf.reshape(x,[-1,28,28,1])
#we will x_image and weight vector to convolution ,plus bias then
#apply the Relu function  final use max pooling
h_conv1=tf.nn.relu((conv2d(x_image,W_conv1)+b_conv1))
h_pool1=max_pooling_2x2(h_conv1)
#second convolution layer
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pooling_2x2(h_conv2)
#now image reduce 7*7,we join a have 1024 Neuron full connection layer,then deal with image.
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#for reduce overfitting,we join dropout before outer layer
keep_prob=tf.placeholder('float')
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#finnaly,join softmax layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#train and optimization model


