import tensorflow as tf
import numpy as np

input_dim=2
output_dim=1

x = tf.placeholder("float", [None, input_dim])
W = tf.Variable(tf.random_uniform([input_dim, output_dim], -1.0, 1.0))
b = tf.Variable(tf.random_normal([output_dim]))
y = tf.nn.sigmoid(tf.matmul(x, W) + b)

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
       [0.],
       [0.],
       [1.]])

   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
   print (i)
   print (sess.run(y, feed_dict={x: batch_xs, y_: batch_ys}))
