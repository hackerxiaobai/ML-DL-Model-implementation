import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1,1,1,1],padding='VALID'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


def max_pool_2x2(x,ksize=[1,1,1,1],strides=[1,1,1,1],padding='VALID'):
    return tf.nn.max_pool(x, ksize=ksize,strides=strides, padding=padding)


x = tf.placeholder(tf.float32, [None, 227,227,3])
y = tf.placeholder(tf.float32, [None, 1000])

# 第一层卷积层
W_conv1 = weight_variable([11,11, 3, 96])
b_conv1 = bias_variable([96])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1, [1,4,4,1], 'VALID') + b_conv1)
h_pool1 = max_pool_2x2(h_conv1,[1,3,3,1],[1,2,2,1],'VALID')

# 第二层卷积层
W_conv2 = weight_variable([5, 5, 96, 256])
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, [1,1,1,1], 'SAME') + b_conv2)
h_pool2 = max_pool_2x2(h_conv2,[1,3,3,1],[1,2,2,1],'VALID')

# 第三层卷积层
W_conv3 = weight_variable([3, 3, 256, 384])
b_conv3 = bias_variable([384])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, [1,1,1,1], 'SAME') + b_conv3)

# 第四层卷积层
W_conv4 = weight_variable([3, 3, 384, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, [1,1,1,1], 'SAME') + b_conv4)
h_pool4 = max_pool_2x2(h_conv4,[1,3,3,1],[1,2,2,1],'VALID')

# 全连接层，输出为4096维的向量
W_fc1 = weight_variable([6 * 6 * 256, 4096])
b_fc1 = bias_variable([4096])
h_pool4_flat = tf.reshape(h_pool4, [-1, 6 * 6 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
# 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
keep_prob1 = tf.placeholder(tf.float32)
h_fc1_drop1 = tf.nn.dropout(h_fc1, keep_prob1)

# 全连接层，输出为4096维的向量
W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop1, W_fc2) + b_fc2)
# 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
keep_prob2 = tf.placeholder(tf.float32)
h_fc1_drop2 = tf.nn.dropout(h_fc2, keep_prob2)

# 全连接层，输出为1000维的向量
W_fc3 = weight_variable([4096, 1000])
b_fc3 = bias_variable([1000])
output = tf.nn.softmax(tf.matmul(h_fc1_drop2, W_fc3) + b_fc3)

# 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
# 同样定义train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义测试的准确率
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建Session和变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


data = np.random.random((1, 227, 227, 3))
labels = np.random.randint(2, size=(1, 1))
y_binary = to_categorical(labels, 1000)


# 训练200步
for i in range(200):
    # 每10步报告一次在验证集上的准确度
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: data, y: y_binary, keep_prob1: 1.0, keep_prob2:1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: data, y: y_binary, keep_prob1: 0.5,keep_prob2:0.5})

# 训练结束后报告在测试集上的准确度
# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))