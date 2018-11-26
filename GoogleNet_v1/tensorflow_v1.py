import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv1x1(x,in_c,out_c):
	return tf.nn.relu(tf.nn.conv2d(x,weight_variable([1,1,in_c,out_c]),[1,1,1,1], 'VALID')+
						bias_variable([out_c]))

def conv3x3(x,in_c,out_c):
	return tf.nn.relu(tf.nn.conv2d(x,weight_variable([3,3,in_c,out_c]),[1,1,1,1], 'SAME')+
						bias_variable([out_c]))

def conv5x5(x,in_c,out_c):
	return tf.nn.relu(tf.nn.conv2d(x,weight_variable([5,5,in_c,out_c]),[1,1,1,1], 'SAME')+
						bias_variable([out_c]))

def inception_module(in_tensor, c1, c3_1, c3, c5_1, c5, pp):
	conv1 = conv1x1(in_tensor,in_tensor.shape[-1].value,c1)
	conv3_1 = conv1x1(in_tensor,in_tensor.shape[-1].value,c3_1)
	conv3 = conv3x3(conv3_1,conv3_1.shape[-1].value,c3)
	conv5_1 = conv1x1(in_tensor,in_tensor.shape[-1].value,c5_1)
	conv5 = conv5x5(conv5_1,conv5_1.shape[-1].value,c5)
	pool_conv = conv1x1(in_tensor,in_tensor.shape[-1].value,pp)
	pool = tf.nn.max_pool(pool_conv, [1,3,3,1], [1,1,1,1], 'SAME')
	merged = tf.concat([conv1, conv3, conv5, pool],-1)
	return merged

def aux_clf(in_tensor):
	avg_pool = tf.nn.avg_pool(in_tensor, [1,5,5,1], [1,3,3,1], 'VALID')
	conv = conv1x1(avg_pool, avg_pool.shape[-1].value, 128)
	dim = conv.shape[1].value*conv.shape[2].value*conv.shape[3].value
	flattened = tf.reshape(conv,  [-1, dim])

	W_fc1 = weight_variable([dim, 1024])
	b_fc1 = bias_variable([1024])
	h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)
	# 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
	global keep_prob1
	keep_prob1 = tf.placeholder(tf.float32)
	h_fc1_drop1 = tf.nn.dropout(h_fc1, keep_prob1)

	W_fc2 = weight_variable([1024, 1000])
	b_fc2 = bias_variable([1000])
	out = tf.nn.relu(tf.matmul(h_fc1_drop1, W_fc2) + b_fc2)
	return out


def inception_net(in_shape=(224,224,3), n_classes=1000):

	x = tf.placeholder(tf.float32, [None, 224,224,3])
	y = tf.placeholder(tf.float32, [None, 1000])

	# 第一层卷积层
	W_conv1 = weight_variable([7, 7, 3, 64])
	b_conv1 = bias_variable([64])
	h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, [1,2,2,1], 'SAME') + b_conv1)
	pad1 = tf.keras.layers.ZeroPadding2D()(h_conv1)
	pool1 = tf.nn.max_pool(pad1, [1,3,3,1], [1,2,2,1], 'VALID')

	conv2_1 = conv1x1(pool1,pool1.shape[-1].value,64)
	conv2_2 = conv3x3(conv2_1,conv2_1.shape[-1].value,192)

	pad2 = tf.keras.layers.ZeroPadding2D()(conv2_2)
	pool2 = tf.nn.max_pool(pad2, [1,3,3,1], [1,2,2,1], 'VALID')
	inception3a = inception_module(pool2, 64, 96, 128, 16, 32, 32)
	inception3b = inception_module(inception3a, 128, 128, 192, 32, 96, 64)
	pad3 = tf.keras.layers.ZeroPadding2D()(inception3b)
	pool3 = tf.nn.max_pool(pad3, [1,3,3,1], [1,2,2,1], 'VALID')
	inception4a = inception_module(pool3, 192, 96, 208, 16, 48, 64)
	inception4b = inception_module(inception4a, 160, 112, 224, 24, 64, 64)
	inception4c = inception_module(inception4b, 128, 128, 256, 24, 64, 64)
	inception4d = inception_module(inception4c, 112, 144, 288, 32, 48, 64)
	inception4e = inception_module(inception4d, 256, 160, 320, 32, 128, 128)
	pad4 = tf.keras.layers.ZeroPadding2D()(inception4e)
	pool4 = tf.nn.max_pool(pad4, [1,3,3,1], [1,2,2,1], 'VALID')
	aux_clf1 = aux_clf(inception4a)
	aux_clf2 = aux_clf(inception4d)
	inception5a = inception_module(pool4, 256, 160, 320, 32, 128, 128)
	inception5b = inception_module(inception5a, 384, 192, 384, 48, 128, 128)
	pad5 = tf.keras.layers.ZeroPadding2D()(inception5b)
	pool5 = tf.nn.max_pool(pad5, [1,3,3,1], [1,2,2,1], 'VALID')
	# 全局平均池化
	avg_pool_1 = tf.nn.avg_pool(pool5,ksize=[1,4,4,1],strides=[1,1,1,1],padding='VALID')
	avg_pool = tf.reshape(avg_pool_1, [-1, 1*1*1024])
	# 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
	keep_prob2 = tf.placeholder(tf.float32)
	h_fc1_drop2 = tf.nn.dropout(avg_pool, keep_prob2)
	W_fc3 = weight_variable([1024, 1000])
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

	data = np.random.random((2, 224, 224, 3))
	labels = np.random.randint(2, size=(2, 1))
	y_binary = to_categorical(labels, 1000)


	# 训练200步
	for i in range(200):
	    # 每10步报告一次在验证集上的准确度
	    if i % 10 == 0:
	        train_accuracy = accuracy.eval(feed_dict={x: data, y: y_binary, keep_prob1: 1.0, keep_prob2:1.0})
	        print("step %d, training accuracy %g" % (i, train_accuracy))
	    sess.run([train_step],feed_dict={x: data, y: y_binary, keep_prob1: 0.5,keep_prob2:0.5})

	# 训练结束后报告在测试集上的准确度
	# print("test accuracy %g" % accuracy.eval(feed_dict={
	#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
if __name__ == '__main__':
	inception_net()