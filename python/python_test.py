import tensorflow as tf

x = tf.constant(1,shape=[2,3])
with tf.Session() as sess:
    print(sess.run(x))
    sum = tf.reduce_sum(x)
    print(sess.run(sum))
    sum1 = tf.reduce_sum(x,1,keep_dims=True)
    print(sess.run(sum))
