import tensorflow as tf
import numpy as np

matrix1 = tf.placeholder(dtype=tf.float32,shape=[100,10])
matrix2 = tf.placeholder(dtype=tf.float32,shape=[10,100])

with tf.Session() as sess:
    product = tf.matmul(matrix1,matrix2)
    result = sess.run(product,feed_dict={
                matrix1:np.random.randn(100,10),
                matrix2:np.random.randn(10,100)})
    print(result)