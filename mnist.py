#INTEL DEEP LEARNING EVENT 

#Reads and downloads the data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)

import tensorflow as tf

#value to input 
x = tf.placeholder(tf.float32, [None, 784])

#weight (w) and bias (b) for our model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#placeholder for the predicted answers
y = tf.nn.softmax(tf.matmul(x, W) + b)
#placeholder to input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

#calculating the error margin
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#minimizing cross entropy using gradient descent algorithm and 0.5 learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#launch the model in an interactive session
sess = tf.InteractiveSession()
#initialize all variables created
tf.global_variables_initializer().run()
#Run the training step 1000 times
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  #evaluating how well the model did giving a list of booleans
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #ask for the accuracy on the dataset
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))