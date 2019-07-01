#https://baike.baidu.com/item/AlexNet/22689612?fr=aladdin
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data',one_hot=True)
learning_rate=0.001
training_iters=200000
batch_size=64
display_step=20
n_input=784
n_classes=10
dropout=0.8
x=tf.placeholder(tf.dtypes.float32,[None,n_input])
y=tf.placeholder(tf.dtypes.float32,[None,n_classes])
keep_prob=tf.placeholder(tf.dtypes.float32)
#juanjicaozuo
def connv2d(name,l_input,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input,w,strides=[1,1,1,1],padding='SAME'),b),name=name)
#max_low_pooling_operate
def max_pool(name,l_input,k):
    return tf.nn.max_pool(l_input,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)
#operate_normalization
def norm(name,l_input,lsize=4):
    return tf.nn.lrn(l_input,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75,name=name)
#define the entire network
def alex_net(_X,_weights,_biases,_dropout):
    _X=tf.reshape(_X,shape=[-1,28,28,1])#Vector to Matrix
    #Convolution layer
    conv1=connv2d('conv1',_X,_weights['wc1'],_biases['bc'])
    #down sampling layer
    pool1=max_pool('pool1',conv1,k=2)
    #normalization layer
    norm1=norm('norm1',pool1,lsize=4)
    #Dropout
    norm1=tf.nn.dropout(norm1,dropout)
    #convoluion
    conv2=connv2d('conv2',norm1,_weights['wc2'],_biases['bc2'])
    #Down sampleing
    pool2=max_pool('pool2',conv2,k=2)
    #normalization
    norm2=norm('norm2',pool2,lsize=4)
    #Dropout
    norm2=tf.nn.dropout(norm2,dropout)
    #convolution
    conv3=connv2d('conv3',norm2,_weights['wc3'],_biases['bc3'])
    #down sampleing
    pool3=max_pool('pool3',conv3,k=2)
    #normalization
    norm3=norm('norm3',pool3,lsize=4)
    #dropout
    norm3=tf.nn.dropout(norm3,_dropout)

    #fully connection layer
    #Firstly,the feature graph is transformed into a vector
    dense1=tf.reshape(norm3,[-1,_weights['wd1'].get_shape().as_list()[0]])
    dense1=tf.nn.relu(tf.matmul(dense1,_weights['wd1'])+_biases['bd1'],name='fc1')
    #fully connection layer
    dense2=tf.nn.relu(tf.matmul(dense1,_weights['wd2'])+_biases['bd2'],name='fc2')
    #Relu activation
    #net output layer
    out=tf.matmul(dense2,_weights['out'])+_biases['out']
    return out
#store all net_parameter
weights={
    'wc1':tf.Variable(tf.random_normal([3,3,1,64])),
    'wc2':tf.Variable(tf.random_normal([3,3,64,128])),
    'wc3':tf.Variable(tf.random_normal([3,3,128,256])),
    'wd1':tf.Variable(tf.random_normal([4*4*256,1024])),
    'wd2':tf.Variable(tf.random_normal([1024,1024])),
    'out':tf.Variable(tf.random_normal([1024,10]))
}
biases={
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
#build model
pred=alex_net(x,weights,biases,keep_prob)
#Define loss function and learning steps
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimzer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#test net
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.case(correct_pred,tf.float32))
#initialize all shared variables
init=tf.initialize_all_variables()
#open a train
with tf.Session() as sess:
    sess.run(init)
    step=1
    #Keep training until reach max iterations
    while step*batch_size <training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        #get new data
        sess.run(optimzer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:dropout})
        if step%display_step==0:
            #computation accuracy
            acc=sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            #computation loss
            loss=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            print('Iter'+str(step*batch_size)+',Minibatch_loss='+'{:.6f}'.format(loss)+',Training Accuracy= '+'{:.5f}'.format(acc))
        step+=1
    print('Optimization Finished')
    #computation_test_accuracy
    print('Testing Accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.}))



