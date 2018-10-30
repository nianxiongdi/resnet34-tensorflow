

import tensorflow as tf
import tensorflow.contrib.slim as slim


from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
import matplotlib.pyplot as plt


# 读取文件
def load_pickle(f):
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == '2':
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


'''
#测试数据

if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_CIFAR10('./cifar-10-batches-py')

    print(train_img.shape)
    print(train_label.shape)

    print(test_img.shape)
    print(test_label.shape)
'''




#双层残差模块
def res_layer2d(input_tensor,kshape = [5,5],deph = 64,conv_stride = 1,padding='SAME'):
    data = input_tensor

    data = slim.batch_norm(data, activation_fn=tf.nn.relu)

    #模块内部第一层卷积
    data = slim.conv2d(data,num_outputs=deph,kernel_size=kshape,stride=conv_stride,padding=padding)

    # #模块内部第二层卷积
    data = slim.conv2d(data,num_outputs=deph,kernel_size=kshape,stride=conv_stride,padding=padding,activation_fn=None)
    output_deep = input_tensor.get_shape().as_list()[3]

    #当输出深度和输入深度不相同时，进行对输入深度的全零填充
    if output_deep != deph:
        input_tensor = tf.pad(input_tensor,[[0, 0], [0, 0], [0, 0],[abs(deph-output_deep)//2,abs(deph-output_deep)//2] ])
    data = tf.add(data,input_tensor)
    data = tf.nn.relu(data)
    return data




#模型在增加深度的同时，为了减少计算量进行的xy轴降维（下采样），
# #这里用卷积1*1，步长为2。当然也可以用max_pool进行下采样，效果是一样的
def get_half(input_tensor,deph):
    data = input_tensor
    data = slim.conv2d(data,deph//2,1,stride = 2)
    return data

#组合同类残差模块
def res_block(input_tensor,kshape,deph,layer = 0,half = False,name = None):
    data = input_tensor
    with tf.variable_scope(name):
        if half:
            data = get_half(data,deph//2)
        for i in range(layer//2):
            data = res_layer2d(input_tensor = data,deph = deph,kshape = kshape)
        return data




CONV_SIZE = 3
CONV_DEEP = 64
NUM_LABELS = 10


#定义模型传递流程
def inference(input_tensor, regularizer = None):
    with slim.arg_scope([slim.conv2d,slim.max_pool2d],stride = 1,padding = 'SAME'):

        with tf.variable_scope("layer1-initconv"):

            data = slim.conv2d(input_tensor, CONV_DEEP , [7, 7])
            data = slim.max_pool2d(data,[2,2],stride=2)

            with tf.variable_scope("resnet_layer"):

                data = res_block(input_tensor = data,kshape = [CONV_SIZE, CONV_SIZE],deph = CONV_DEEP,layer = 6,half = False,name = "layer4-9-conv")
                data = res_block(input_tensor = data,kshape = [CONV_SIZE, CONV_SIZE],deph = CONV_DEEP * 2,layer = 8,half = True,name = "layer10-15-conv")
                data = res_block(input_tensor = data,kshape = [CONV_SIZE, CONV_SIZE],deph = CONV_DEEP * 4,layer = 12,half = True,name = "layer16-27-conv")
                data = res_block(input_tensor = data,kshape = [CONV_SIZE, CONV_SIZE],deph = CONV_DEEP * 8,layer = 6,half = True,name = "layer28-33-conv")

                data = slim.avg_pool2d(data,[2,2],stride=2)

                #得到输出信息的维度，用于全连接层的输入
                data_shape = data.get_shape().as_list()
                nodes = data_shape[1] * data_shape[2] * data_shape[3]
                reshaped = tf.reshape(data, [data_shape[0], nodes])
                #最后全连接层
                with tf.variable_scope('layer34-fc'):
                    fc_weights = tf.get_variable("weight", [nodes, NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))

                    # if regularizer != None:
                    #     tf.add_to_collection('losses', regularizer(fc_weights))

                    fc_biases = tf.get_variable("bias", [NUM_LABELS],initializer=tf.constant_initializer(0.1))
                    fc = tf.nn.relu(tf.matmul(reshaped, fc_weights) + fc_biases)

                    # if train:
                    #     fc = tf.nn.dropout(fc, 0.5)
                    #     return fc

                    return fc

def conputer_loss(pre, label):

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pre))

    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    return loss, train_op


# from datetime import datetime
# import math
# import time
#
# # 评测函数
# def time_tensorflow_run(session, target, info_string):
#     num_steps_burn_in = 10
#     total_duration = 0.0
#     total_duration_squared = 0.0
#     for i in range(num_batches + num_steps_burn_in):
#         start_time = time.time()
#         _ = session.run(target)
#         duration = time.time() - start_time
#         if i >= num_steps_burn_in:
#             if not i % 10:
#                 print('%s: step %d, duration = %.3f' %
#                       (datetime.now(), i - num_steps_burn_in, duration))
#             total_duration += duration
#             total_duration_squared += duration * duration
#     mn = total_duration / num_batches
#     vr = total_duration_squared / num_batches - mn * mn
#     sd = math.sqrt(vr)
#     print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
#           (datetime.now(), info_string, num_batches, mn, sd))
#
#
# batch_size = 6
# height, width = 240, 320
# inputs = tf.random_uniform((batch_size, 32, 32, 3))
# print(type(inputs))
# print(inputs.get_shape())
# net = inference(inputs, 6) # 152层评测
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# num_batches = 100
# time_tensorflow_run(sess, net, "Forward")



batch_size = 64
learn_rate = 1e-4
WIDTH = 32
HEIGHT = 32



if __name__ == '__main__':

    inputs = tf.placeholder(shape=[batch_size, WIDTH, HEIGHT, 3],dtype=tf.float32)
    labels = tf.placeholder(shape=[batch_size, ],dtype=tf.int64)

    label  = tf.one_hot(labels, NUM_LABELS)

    pre = inference(input_tensor=inputs)
    pre = tf.nn.softmax(pre)

    pre1 = tf.argmax(pre, axis=1)

    acc = tf.equal(x=pre1, y=labels)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    loss, train_op = conputer_loss(pre=pre, label=label)

    train_img, train_label, test_img, test_label = load_CIFAR10('./cifar-10-batches-py')


    import  time
    import random

    with tf.Session() as sess:

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./output/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        index = [i for i in range(50000)]
        for epoch in range(3000):

            start = 0

            total_loss = 0.0
            total_acc = 0.0



            random.shuffle(index)
            images_data = train_img[index]
            images_label = train_label[index]

            for i in range(50000//batch_size - 1):
                end = start + batch_size
                # print(start, end)

                images_data = train_img[start:end]
                images_label = train_label[start:end]
                # print(images_data.shape)
                # print(images_label.shape)
                # print(images_label)
                images_data = images_data / 255
                loss1 , _ , acc1= sess.run([loss, train_op,acc], feed_dict={inputs:images_data, labels:images_label})

                # print(pre1[0])
                # print(loss1)

                total_loss += loss1
                total_acc +=acc1
                if(i % 200 == 0 ):

                    if i  == 0 :
                        print('epoch:', epoch, ' batch:', i, 'loss:', total_loss , ' acc:', total_acc )
                    else:
                        print('epoch:', epoch, ' batch:', i, 'loss:', total_loss / 200, ' acc:', total_acc / 200)


                    total_loss = 0.0
                    total_acc = 0.0


                # print(pre1.shape)
                # print(loss1)
                # loss1, _ = sess.run([loss, train_op], feed_dict={inputs:images_data, labels:images_label})
                # print(loss1)
                start = end

            saver.save(sess, './output/', global_step=epoch)












