import argparse
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import PhasedLSTMCell
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn import dynamic_rnn

##from helpers.file_logger import FileLogger


#  输出神经元数量
num_classes = 10
# 输入神经元数量
mnist_img_size = 28 * 28


lstm_cell=PhasedLSTMCell  
hidden_size=32  # 隐含层的神经元数量
batch_size=256  # 每一批次的 样本量
steps=10  #  迭代次数
log_file=r'C:\working\时序数据\log.tsv'  #  迭代信息记录文件

#  数据文件位置
datafile=r'C:\working\时序数据\tensorflow-phased-lstm-master\MNIST_data'


# 读取数据
mnist = input_data.read_data_sets(datafile, one_hot=True)

# 设置学习速率，梯度下降速率参数
learning_rate = 0.001

# 创建并连接记录文本文件，写入标题行  --  下面的列表
##file_logger = FileLogger(log_file, ['step', 'training_loss', 'training_accuracy'])




#  构建神经元

#  x_  batch_size 是每一批的样本量，minist_img_size 是 输入神经元数量 ，1 代表每个变量单独输入
x_ = tf.placeholder(tf.float32, (batch_size, mnist_img_size, 1))


#  t_  batch_size 是每一批的样本量，minist_img_size 是 输入神经元数量 ，1 代表每个变量单独输入
t_ = tf.placeholder(tf.float32, (batch_size, mnist_img_size, 1))


#  y_  batch_size 是每一批的数量， num_classes 是输出神经元数量
y_ = tf.placeholder(tf.float32, (batch_size, num_classes))

if lstm_cell == PhasedLSTMCell: # 这里替换成  BasicLSTMCell 就变成了普通的LSTM网络
    inputs = (t_, x_)
else:
    inputs = x_

# 由于 LSTM本质上还是 RNN，所以采用dynamic_rnn函数 对 网络进行更新
#
##  def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
##                dtype=None, parallel_iterations=None, swap_memory=False,
##                time_major=False, scope=None):
##         .......
##
##    
##
##
##

outputs, _ = dynamic_rnn(cell=lstm_cell(hidden_size), inputs=inputs, dtype=tf.float32)


rnn_out = tf.squeeze(outputs[:, -1, :])

y = slim.fully_connected(inputs=rnn_out,
                         num_outputs=num_classes,
                         activation_fn=None)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))



grad_update = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

sess.run(tf.global_variables_initializer())

def feed_dict_phased_lstm(batch):
    
    img = np.expand_dims(batch[0], axis=2)
    
    t = np.reshape(np.tile(np.array(range(mnist_img_size)), (batch_size, 1)),\
                       (batch_size, mnist_img_size, 1))
    
    return {x_: img, y_: batch[1], t_: t}

def feed_dict_basic_lstm(batch):
    
    img = np.expand_dims(batch[0], axis=2)
    
    return {x_: img, y_: batch[1]}
##
##for i in range(steps):
##    
##    b = mnist.train.next_batch(batch_size)
##    
##    st = time()
##
##    if lstm_cell == PhasedLSTMCell:
##        
##        feed_dict = feed_dict_phased_lstm(b)
##        
##    else:
##        
##        feed_dict = feed_dict_basic_lstm(b)
##
##    tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, grad_update], \
##                                  feed_dict=feed_dict)
##    
##    print('steps = {0} | time {1:.2f} | tr_loss = {2:.3f} | tr_acc = {3:.3f}'.format(str(i).zfill(6),
##                                                                                     time() - st,
##                                                                                     tr_loss,
##                                                                                     tr_acc))
##    file_logger.write([i, tr_loss, tr_acc])
##
##file_logger.close()
