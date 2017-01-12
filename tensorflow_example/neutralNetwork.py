import tensorflow_self as tf
from tensorflow_self.examples.tutorials.mnist import input_data

###some is error ,remember to solve it !!!
class neutralNetwork():
    def __init__(self,l1_number,l2_number,l3_number,n_class=10,batch_size = 100):
        self.node_h1_number = l1_number
        self.node_h2_number = l2_number
        self.node_h3_number = l3_number
        self.l1_bias = l1_number
        self.l2_bias = l2_number
        self.l3_bias = l3_number
        self.n_class = n_class
        self.batch_size = batch_size
        self.x = tf.placeholder('float', [None, 784])
        self.y = tf.placeholder('float')
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    def neutral_model(self):

        hidden_1_layer = {"weights":tf.Variable(tf.random_normal([self.node_h1_number,784])),
                          "bias":tf.Variable(tf.random_normal([self.l1_bias]))}
        hidden_2_layer = {"weights":tf.Variable(tf.random_normal([self.node_h2_number,self.node_h1_number])),
                          "bias":tf.Variable(tf.random_normal([self.node_h2_number]))}
        hidden_3_layer = {"weights":tf.Variable(tf.random_normal([self.node_h3_number,self.node_h2_number])),
                          "bias":tf.Variable(tf.random_normal([self.node_h3_number]))}
        output_layer = {"weights":tf.Variable(tf.random_normal([self.n_class,self.node_h3_number])),
                        "bias":tf.Variable(tf.random_normal([self.n_class]))}
        l1 = tf.add(tf.matmul(hidden_1_layer["weights"],self.x),hidden_1_layer["bias"])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(hidden_2_layer["weights"],l1),hidden_2_layer["bias"])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(hidden_3_layer["weights"],l2),hidden_3_layer["bias"])
        l3 = tf.nn.relu(l3)

        output = tf.add(tf.matmul(output_layer["weights"],l3),output_layer["bias"])

        return output

    def train_neutral_network(self):
        prediction = self.neutral_model()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        echos = 100
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for echo in range(echos):
                loss = 0
                for _ in range(int(self.mnist.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y = self.mnist.train.next_batch(self.batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                loss += c

                print('Epoch', echo, 'completed out of', echos, 'loss:', loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({self.x: self.mnist.test.images, self.y: self.mnist.test.labels}))

neutral = neutralNetwork(500,500,500)
neutral.train_neutral_network()