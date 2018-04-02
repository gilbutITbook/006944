import tensorflow as tf
import numpy as np

input_dim=2
hidden_dim=2
output_dim=1

x = tf.placeholder("float", [None, input_dim])
W1 = tf.Variable(tf.random_uniform([input_dim, hidden_dim], -1.0, 1.0))
b1 = tf.Variable(tf.random_normal([hidden_dim]))
W2 = tf.Variable(tf.random_uniform([hidden_dim, output_dim], -1.0, 1.0))
b2 = tf.Variable(tf.random_normal([output_dim]))
layer1 = tf.sigmoid(tf.matmul(x,W1)+b1)
layer2 = tf.matmul(layer1,W2)+b2
y = layer2

y_ = tf.placeholder("float", [None, output_dim])
loss = tf.reduce_mean(tf.square(y - y_))

train_step = tf.train.MomentumOptimizer(0.01,0.97).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(5000):
   batch_xs = np.array([
       [0., 0.],
       [0., 1.],
       [1., 0.],
       [1., 1.]])
   batch_ys =  np.array([
       [0.],
       [1.],
       [1.],
       [0.]])

sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print i,sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})

