"""
    This python file mainly trains the model based on MNIST and 
    generates a prediction.txt file of test example images containing 
    all the predictions from the trained model, then calculates the 
    accuracy.
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import argparse
import os
from sys import argv
from glob import glob
from scipy import misc
import numpy
import ntpath
import skimage.io
import skimage.transform
import skimage.util
from skimage import exposure

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# 1-->train ...0-->test
train_flag = 1
if (len(argv)>1):
    train_flag = 0


if train_flag == 1:
    # train part
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(600000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 0.5})

        save_path = saver.save(sess, "./model.ckpt")
        print("train model is saved at: %s" % save_path)

else:
    # test part
    with tf.Session() as sess:
        saver.restore(sess, "./model.ckpt")
        print("Model restored.")

        # pre-processing images, then create prediction.txt based on test result
        path = argv[1]
        path_list = glob(path + "/*.png")

        output = open("prediction.txt", "w")
        for img_path in path_list:
            # find the image file's name
            sps = img_path.split(os.sep)
            image_name = sps[-1]
            # process image
            img = misc.imread(img_path)
            img = exposure.adjust_gamma(img, 0.15)
            (vertical_pixel, horizontal_pixel) = img.shape
            if vertical_pixel > horizontal_pixel:
                vertical_padding = int(round(vertical_pixel * 0.15))
                horizontal_padding = int(round((vertical_pixel * 1.3 - horizontal_pixel) / 2))
                padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
                img = skimage.util.pad(img, padding, 'constant', constant_values=0)
            else:
                horizontal_padding = int(round(horizontal_pixel * 0.15))
                vertical_padding = int(round((horizontal_pixel * 1.3 - vertical_pixel) / 2))
                padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
                img = skimage.util.pad(img, padding, 'constant', constant_values=0)

            img = skimage.transform.resize(img, (28, 28))
            img = numpy.reshape(img, (1, 784))

            results = tf.argmax(y_conv, 1)
            num = results.eval(feed_dict = {x: img , keep_prob: 1.0})

            output.write("%s\t%i\n" % (image_name, num[0]))
        output.close()

        # calculate accuracy
        map = {}
        total = 0
        right = 0
        # read annotation.txt file
        with open("./annotation.txt", "r") as ann:
            for ln in ann:
                image, label = ln.split()
                map[image] = label

        # read prediction.txt file
        with open("./prediction.txt", "r") as pred:
            for ln in pred:
                image, label = ln.split()
                total += 1
                if label == map[image]:
                    right += 1

        result = right / total
        print("%.4f" % result)