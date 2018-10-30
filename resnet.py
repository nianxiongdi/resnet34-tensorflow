
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ResNet V2
# 载入模块、TensorFlow
import collections
import tensorflow as tf

slim = tf.contrib.slim

# 定义Block
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    '''
        # 定义一个namedtuple类型User，并包含name，sex和age属性。
        User = namedtuple('User', ['name', 'sex', 'age'])

        # 创建一个User对象
        user = User(name='kongxx', sex='male', age=21)

    '''
    'A named tuple describing a ResNet block'

# 定义降采样subsample方法
def subsample(inputs, factor, scope=None):

    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

# 定义conv2d_same函数创建卷积层
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """
       卷积层实现,有更简单的写法，这样做其实是为了提高效率
       :param inputs: 输入tensor
       :param num_outputs: 输出通道
       :param kernel_size: 卷积核尺寸
       :param stride: 卷积步长
       :param scope: 节点名称
       :return: 输出tensor
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
# 定义堆叠Blocks函数，两层循环
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net

# 创建ResNet通用arg_scope，定义函数默认参数值
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001, # L2权重衰减速率
                     batch_norm_decay=0.997,  # BN的衰减速率
                     batch_norm_epsilon=1e-5,# BN的epsilon默认1e-5
                     batch_norm_scale=True): # BN的scale默认值

    '''

        'is_training':
            # 是否是在训练模式，如果是在训练阶段，将会使用指数衰减函数（衰减系数为指定的decay），
            # 对moving_mean和moving_variance进行统计特性的动量更新，也就是进行使用指数衰减函数对均值和方
            # 差进行更新,而如果是在测试阶段，均值和方差就是固定不变的，是在训练阶段就求好的，在训练阶段，
            # 每个批的均值和方差的更新是加上了一个指数衰减函数，而最后求得的整个训练样本的均值和方差就是所
            # 有批的均值的均值，和所有批的方差的无偏估计

        'decay': batch_norm_decay,
             # 该参数能够衡量使用指数衰减函数更新均值方差时，更新的速度，取值通常在0.999-0.99-0.9之间，值
            # 越小，代表更新速度越快，而值太大的话，有可能会导致均值方差更新太慢，而最后变成一个常量1，而
            # 这个值会导致模型性能较低很多.另外，如果出现过拟合时，也可以考虑增加均值和方差的更新速度，也
            # 就是减小decay

        'epsilon': batch_norm_epsilon,
            是在归一化时，除以方差时，防止方差为0而加上的一个数

        'scale': batch_norm_scale,

        'updates_collections': tf.GraphKeys.UPDATE_OPS,
             # force in-place updates of mean and variance estimates
            # 该参数有一个默认值，ops.GraphKeys.UPDATE_OPS，当取默认值时，slim会在当前批训练完成后再更新均
            # 值和方差，这样会存在一个问题，就是当前批数据使用的均值和方差总是慢一拍，最后导致训练出来的模
            # 型性能较差。所以，一般需要将该值设为None，这样slim进行批处理时，会对均值和方差进行即时更新，
            # 批处理使用的就是最新的均值和方差。
            #
            # 另外，不论是即使更新还是一步训练后再对所有均值方差一起更新，对测试数据是没有影响的，即测试数
            # 据使用的都是保存的模型中的均值方差数据，但是如果你在训练中需要测试，而忘了将is_training这个值
            # 改成false，那么这批测试数据将会综合当前批数据的均值方差和训练数据的均值方差。而这样做应该是不
            # 正确的。
    '''


    batch_norm_params = {# 定义batch normalization（标准化）的参数字典
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,

    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay), # 权重正则器设置为L2正则
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,   # 标准化器设置为BN
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):

            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
# 定义核心bottleneck残差学习单元
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    """
       核心残差学习单元
       输入tensor给出一个直连部分和残差部分加和的输出tensor
       :param inputs: 输入tensor
       :param depth: Block类参数，输出tensor通道
       :param depth_bottleneck: Block类参数，中间卷积层通道数
       :param stride: Block类参数，降采样步长
                      3个卷积只有中间层采用非1步长去降采样。
       :param outputs_collections: 节点容器collection
       :return: 输出tensor
   """

    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:

        # 获取输入tensor的最后一个维度(通道)
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

        # 对输入正则化处理，并激活
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        # shortcut直连部分
        if depth == depth_in:
            # 如果输入tensor通道数等于输出tensor通道数
            # 降采样输入tensor使之宽高等于输出tensor
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            # 否则，使用尺寸为1*1的卷积核改变其通道数,
            # 同时调整宽高匹配输出tensor
            #  用B方式解决不同维度问题
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        # residual残差部分
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)

# 定义生成ResNet V2的主函数
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    """
       网络结构主函数
       :param inputs: 输入tensor
       :param blocks: Block类列表
       :param num_classes: 输出类别数
       :param global_pool: 是否最后一层全局平均池化
       :param include_root_block: 是否最前方添加7*7卷积和最大池化
       :param reuse: 是否重用
       :param scope: 整个网络名称
       :return:
    """

    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            print(type(inputs))
            if include_root_block:

                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)

            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points['predictions']






# 设计层数为50的ResNet V2
def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

# 设计101层的ResNet V2
def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

# 设计152层的ResNet V2
def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    blocks = [ # 输出深度，瓶颈深度，瓶颈步长
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

# 设计200层的ResNet V2
def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

from datetime import datetime
import math
import time

# 评测函数
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


batch_size = 6
height, width = 240, 320
inputs = tf.random_uniform((batch_size, height, width, 3))
print(type(inputs))
print(inputs.get_shape())
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_50(inputs, 1000) # 152层评测

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "Forward")
