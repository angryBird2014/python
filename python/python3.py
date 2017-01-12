import tensorflow as tf
import numpy as np

A = tf.Variable(tf.random_normal([2,2]),name='MartixA')
B = tf.Variable(tf.random_uniform(shape=[2,2]),name='B')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        tmp = tf.matmul(tf.transpose(A),B)
        print(sess.run(tmp),"ssss",sess.run(A),"ssss",sess.run(B))
        print("----------")
        tf.assign(B,tmp)