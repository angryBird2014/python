import tensorflow as tf
import numpy as np

matrixA = tf.constant(3,shape=[3,4])
matrixB = tf.constant(4,shape=[4,2])

product = tf.matmul(matrixA,matrixB)

with tf.Session() as sess:
    result = sess.run(product)
    result_value = tf.cast(result,np.float32)