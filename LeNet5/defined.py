import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10
IMAGE_SIZE=28
NUM_CHANNEL=1
NUM_LABEL=10
#layer1
CONV1_DEEP=32
CONV1_SIZE=5
#layer2
CONV2_DEEP=64
CONV2_SIZE=5
#full convolution layer
FC_SIZE=512
#LAYER_NODE=500
def interence(input_tensor,train,regularize):
    with tf.variable_scope('layer_conv'):
        w=tf.get_variable('w',[CONV1_SIZE,CONV1_DEEP,NUM_CHANNEL,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b=tf.get_variable('b',shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        #filter shape is [filter_height,filter_width,in_channels,out_channels]
        #input tensor shape is:[batch,in_weight,in_width,in_channels]
        #'strides=[1,stride,stride,1]'
        #return [batch,height,width,channels].
        conv1=tf.nn.conv2d(input_tensor,w,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,b))
    with tf.variable_scope('layer2_pool'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv'):
        w=tf.get_variable('w',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        b=tf.get_variable('b',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,w,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,b))

    with tf.variable_scope('layer4-pool'):
        #pool2 size is [batch_size,7,7,64]
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #full connect layer ,need pool2 change a one-dimensional vector,for input future.
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[20,nodes])
    with tf.variable_scope('layer5-fc1'):
        fc1_w=tf.get_variable('w',shape=[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        try:
            #only full connect layer weight need join regularization
            if regularize !=None:
                tf.add_to_collection('loss',regularize(fc1_w))
        except:
            pass
        fc1_b=tf.get_variable('b',shape=[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
        #use dropout random will many output of node change 0, for avoid overfitting,then let model  training data perfermance better
        #dropout ordinary in full connect layer.
        if train:
            fc1=tf.nn.dropout(fc1,0.5)
    with tf.variable_scope('layer6_fc2'):
        fc2_w=tf.get_variable('w',shape=[FC_SIZE,NUM_LABEL],initializer=tf.truncated_normal_initializer(stddev=0.1))
        try:
            if regularize !=None:
                tf.add_to_collection('loss',regularize(fc2_w))
        except:
            pass
        fc2_b=tf.get_variable('b',shape=[NUM_LABEL],initializer=tf.constant_initializer(0.1))
        #output of endder layer,not need join activate function
        logit=tf.matmul(fc1,fc2_w)+fc2_b
    return logit