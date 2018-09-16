# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:01:20 2018

@author: zuodo
"""

import tensorflow as tf
a=tf.constant([[1,2,3],[2,3,4],[1,2,2]])
b=tf.constant([1,1,1])
c=a+b
a1=tf.constant([[1,2],[3,2]])
b1=tf.constant([[2,3],[4,5]])
c1=a1*b1 
c2=tf.matmul(a1,b1)
sess=tf.Session()
print(sess.run(c),"\n",sess.run(c1),"\n",sess.run(c2))
