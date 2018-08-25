# Imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from plotting import *
import os 

# Path and Dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
mnist = input_data.read_data_sets(dir_path + '/MNIST_data/', one_hot=True)

# Variables
hidden_layer_1_nodes = 500
hidden_layer_2_nodes = 500
output_layer_nodes = 500
epochs = 10
classes = 10
epoch_errors = []
stddev = 0.035
learning_rate = 0.08
batch_size = 50

# TF Placeholders
X = tf.placeholder('float', [None, 784], name='X')
y = tf.placeholder('float', name='y')
# Weights Matrices
W1 = tf.Variable(tf.truncated_normal([784, hidden_layer_1_nodes], stddev=stddev), name='W1')
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1_nodes, hidden_layer_2_nodes], stddev=stddev), name='W2')
W3 = tf.Variable(tf.truncated_normal([hidden_layer_2_nodes, output_layer_nodes], stddev=stddev), name='W3')
W4 = tf.Variable(tf.truncated_normal([output_layer_nodes, classes], stddev=stddev), name='W4')
# Biases Vectors
b1 = tf.Variable(tf.truncated_normal([hidden_layer_1_nodes], stddev=stddev), name='b1')
b2 = tf.Variable(tf.truncated_normal([hidden_layer_2_nodes], stddev=stddev), name='b2')
b3 = tf.Variable(tf.truncated_normal([output_layer_nodes], stddev=stddev), name='b3')
b4 = tf.Variable(tf.truncated_normal([classes], stddev=stddev), name='b4')

# Define the Neural Network
def nn_model(X):
    input_layer     =    {'weights': W1, 'biases': b1}
    hidden_layer_1  =    {'weights': W2, 'biases': b2}
    hidden_layer_2  =    {'weights': W3, 'biases': b3}
    output_layer    =    {'weights': W4, 'biases': b4}

    input_layer_sum = tf.add(tf.matmul(X, input_layer['weights']), 
                            input_layer['biases'])
    input_layer_sum = tf.nn.relu(input_layer_sum)

    hidden_layer_1_sum = tf.add(tf.matmul(input_layer_sum, hidden_layer_1['weights']), 
                            hidden_layer_1['biases'])
    hidden_layer_1_sum = tf.nn.relu(hidden_layer_1_sum)

    hidden_layer_2_sum = tf.add(tf.matmul(hidden_layer_1_sum, hidden_layer_2['weights']), 
                            hidden_layer_2['biases'])
    hidden_layer_2_sum = tf.nn.relu(hidden_layer_2_sum)

    output_layer_sum = tf.add(tf.matmul(hidden_layer_2_sum, output_layer['weights']), 
                            output_layer['biases'])
    return output_layer_sum

# Train the Neural Network
def nn_train(X):
    pred = nn_model(X)
    pred = tf.identity(pred)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        #saver = tf.train.Saver()
        sess.run(init_op)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, y: epoch_y})
                epoch_loss += c
            epoch_errors.append(epoch_loss)
            print('Epoch ', epoch + 1, ' of ', epochs, ' with loss: ', epoch_loss)

        correct_result = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_result, 'float'))
        print('Acc: ', accuracy.eval({X:mnist.test.images, y:mnist.test.labels})) 
        # Save the Model (Weights and Biases) 
        #save_path = saver.save(sess, dir_path + "/data/model.ckpt")
        #print("Model saved in file: %s" % save_path)  

if __name__ == "__main__":
    nn_train(X)