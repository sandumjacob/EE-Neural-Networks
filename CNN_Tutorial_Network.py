import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def run_cnn():
    mnist = input_data.read_data_sets("/tmp/data")
    learning_rate = 0.0001
    epochs = 10
    batch_size = 50


# Define tensors and operations to be evaluated in session, these 3 things are like the basic structure without any
# network defined
# x is the input given by the MINST data, a flat 784 value array
x = tf.placeholder(tf.float32, [None, 784])
# x_shaped is a operation (not a tensor) that is a node in the graph which reshapes the tensor x into
# a 28x28 tensor representing the MNIST image
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
# y is the output tensor defined as a 1d array size 10 representing the 10 classes (0-9=10 different classes)
y = tf.placeholder(tf.float32, [None, 10])



