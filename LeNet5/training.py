import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import defined
import numpy as np
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_TATE=0.0001
MOVING_AVERAGE_DECAY=0.99
TRAIN_STEP=300000
MODEL_PATH='model'
MODEL_NAME='model'

def train(mnist):
    x=tf.placeholder(tf.float32,shape=[None,
                                       defined.IMAGE_SIZE,
                                       defined.IMAGE_SIZE,
                                       defined.NUM_CHANNEL],name='x-input')
    y_=tf.placeholder(tf.float32,shape=[None,defined.OUTPUT_NODE])
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_TATE)
    y=defined.interence(x,True,regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_average=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_average_ops=variable_average.apply(tf.trainable_variables())
    cross_entroy=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entroy_mean=tf.reduce_mean(cross_entroy)
    loss=cross_entroy_mean+tf.add_n(tf.get_collection('loss'))
    print('ok')
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    print(learning_rate)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step, variable_average_ops)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAIN_STEP):
            # because of the neural network input is BATCH_SIZE,IMAGE_SIZE,CHANNEL,
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs, (BATCH_SIZE, defined.IMAGE_SIZE,
                                         defined.IMAGE_SIZE,
                                         defined.NUM_CHANNEL))
            _, loss_value, step, learning_rate = sess.run([train_op, loss, global_step, learning_rate],
                                                          feed_dict={x: reshape_xs, y_: ys})
            if i % 1000 == 0:
                print('After %d step ,loss on train is %g,and learn rate is %g' % (step, loss_value, learning_rate))
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()