import tensorflow as tf
import numpy as np

'''
occur NoneType ?????when caculate loss ,ther is a  reduction_indices=[1]
'''
def add_layer(input_data,in_size,out_size,activate_function=None):

    with tf.name_scope('layer'):
        with tf.name_scope("weight"):
            Weight = tf.Variable(tf.random_normal([in_size,out_size],name='weights'))
        with tf.name_scope('bias'):
            Bias = tf.Variable(tf.zeros([1,out_size],name='bias')+0.1)
        with tf.name_scope('output'):
            output = tf.add(tf.matmul(input_data,Weight),Bias)
        if activate_function is None:
            return output
        else:
            return activate_function(output)


x_data = np.linspace(0,5,300,dtype=np.float32).reshape(300,1)
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

inputX = tf.placeholder(dtype=tf.float32,shape=[None,1],name='inputX')
outputY  = tf.placeholder(dtype=tf.float32,shape=[None,1],name='outputY')

l1 = add_layer(inputX,1,10,activate_function=tf.nn.relu)

l2 = add_layer(l1,10,1)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(l2-outputY),reduction_indices=[1]))
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    for i in range(5000):
        sess.run(optimizer, feed_dict={inputX: x_data, outputY: y_data})
        if i % 50 == 0:
           print(sess.run(loss,feed_dict={inputX:x_data,outputY:y_data}))
