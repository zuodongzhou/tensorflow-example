# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:11:04 2018

@author: zuodo
"""
#两个卷积层和一个全连接层
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()
#定义初始化函数创建权重和偏置
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

    #定义卷积层、池化层函数
def conv2D(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    #横竖大小2，横竖步长2
    
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

#将784形式转成28*28，-1代表样本数量不固定，1代表颜色通道1
x_image=tf.reshape(x,[-1,28,28,1])

#定义第一个卷积层
W_conv1=weight_variable([5,5,1,32])#卷积核尺寸5*5，一个颜色通道，32个卷积核
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2D(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#定义第二个卷积层
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2D(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#进行两次池化后边长只有1/4，图片尺寸变为7*7，第二个池化后输出尺寸7*7*64，对其输出变形
#MLP
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#dropout
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#softmax
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#训练
tf.global_variables_initializer().run()
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%1000==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images[:5000],y_:mnist.test.labels[:5000],keep_prob:1.0}))
