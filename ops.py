from unicodedata import name
import numpy as np
import tensorflow as tf
import math
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
from scipy import linalg

def batch_norm(opts, _input, is_train, reuse, scope, scale=True):
    """Batch normalization based on tf.contrib.layers.
    """
    return tf.contrib.layers.batch_norm(
        _input, center=True, scale=scale,
        epsilon=opts['batch_norm_eps'], decay=opts['batch_norm_decay'],
        is_training=is_train, reuse=reuse, updates_collections=None,
        scope=scope, fused=False)

def linear(opts, input_, output_dim, scope=None, init='normal', reuse=None):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()

    assert len(shape) > 0
    in_shape = shape[1]
    if len(shape) > 2:
        input_ = tf.reshape(input_, [-1, np.prod(shape[1:])])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope or "lin", reuse=reuse):
        if init == 'normal':
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        else:
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.constant_initializer(np.identity(in_shape)))
        bias = tf.get_variable(
            "b", [output_dim],
            initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_, matrix) + bias

def conv3d(opts, input_, output_dim, d_d=1, d_h=1, d_w=1, scope=None,
           conv_filters_dim=None, padding='SAME', l2_norm=False):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
        k_h = conv_filters_dim
        k_w = k_h
        k_d = k_w
    else:
        k_h = conv_filters_dim[0]
        k_w = conv_filters_dim[1]
        k_d = conv_filters_dim[2]
    assert len(shape) == 5, 'Conv3d works only with 5d tensors.'
    with tf.variable_scope(scope or 'conv3d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, k_d, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))##
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))##
        conv = tf.nn.bias_add(conv, biases)
    return conv


def deconv3d(opts, input_, output_shape, d_h=1, d_w=1, d_d=1, scope=None, conv_filters_dim=None, padding='SAME'):
    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
        k_h = conv_filters_dim
        k_w = k_h
        k_d = k_w
    else:
        k_h = conv_filters_dim[0]
        k_w = conv_filters_dim[1]
        k_d = conv_filters_dim[2]

    assert len(shape) == 5, 'Conv3d_transpose works only with 5d tensors.'
    assert len(output_shape) == 5, 'outut_shape should be 5dimensional'

    with tf.variable_scope(scope or "deconv3d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, k_d, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv3d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, d_d, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

    return deconv

def conv2d(opts, input_, output_dim, d_h=1, d_w=1, scope=None,
           conv_filters_dim=None, padding='VALID', l2_norm=False):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv2d = tf.nn.bias_add(conv, biases)

    return conv2d

def deconv2d(opts, input_, output_shape, d_h=1, d_w=1, scope=None, conv_filters_dim=None, padding='VAlID'):
    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
    return deconv

def reparameter1(opts,gan_z,cls_cnt,augrate,scope):
    with tf.variable_scope(scope or "reparameter"):
        mu = tf.get_variable('mean_ph',[opts['n_classes'], opts['zdim']],
                             initializer=tf.random_uniform_initializer(-1.,1.),
                             trainable=True)
        sigma = tf.get_variable('sigma_ph', [opts['n_classes'], opts['zdim']],
                             initializer=tf.constant_initializer(5.),
                             trainable=True)
        # repeat_num = tf.get_variable('repeat_num',[1,opts['n_classes']],
        #                     initializer=tf.random_uniform_initializer(1,9),
        #                      trainable=True)
        # shape = sample_points.get_shape().as_list()
        # repeat_num = shape[0]//opts['n_classes']
        # mu = repeat_elements(mu,1,0)
        # sigma = repeat_elements(sigma, 1, 0)
        z = mu + gan_z * sigma
        z1 = repeat_elements(z,cls_cnt,0)
        z2 = repeat_elements(z, np.repeat(int(opts['zdim']*augrate),opts['n_classes']), 0)
    return z1,z2

def augment_batch(opts, x, gan_z, augrate,class_cnt):
    n_classes = len(class_cnt)
    x_aug_list = []
    y_aug_list = []
    aug_nums = [(int(opts['zdim']* augrate) - class_cnt[i]) for i in range(n_classes)]
    gan_begin = 0
    for i in range(n_classes):
        cls_begin = sum(class_cnt[:i])
        if aug_nums[i] <= 0:
            gan_begin=gan_begin+opts['zdim']
            x_aug_list.append(x[cls_begin:cls_begin+int(opts['zdim']*augrate)])
            y_aug_list.append(np.repeat(i, repeats=int(opts['zdim']*augrate)))
            continue
        else:
            tempx = gan_z[gan_begin : gan_begin+class_cnt[i], :]
            if aug_nums[i] > class_cnt[i]:
                for j in range(int(aug_nums[i]-class_cnt[i])):
                    tempx=tf.concat([tempx,tempx[j,tf.newaxis,:]],axis=0)
            else:
                tempx=tempx[0:aug_nums[i],:]
            x_aug_list.append(tf.concat([x[cls_begin:cls_begin+class_cnt[i]], tempx],0))
            y_aug_list.append(np.repeat(i, repeats=int(opts['zdim']* augrate)))
            gan_begin = gan_begin+class_cnt[i]
    aug = tf.concat(x_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    return aug, y_aug #, tr_fa

def schimit(opts, encode_schimit, scope, reuse,cls_cnt):
    related_samples = []
    with tf.variable_scope(scope, reuse=reuse):
        num = 0
        for classes, j in enumerate(cls_cnt):
            if j > opts['zdim']:
                random_num = tf.get_variable('schimit_ph%d' % (classes), [opts['zdim'], opts['zdim']],
                                             initializer=tf.random_uniform_initializer(-1., 1.),trainable=True)
                for i in range(opts['zdim']):
                    tmp = random_num[i, :]
                    tmp = tf.tile(tmp[:, np.newaxis], [1, opts['zdim']])
                    tmp = tf.multiply(tmp, encode_schimit[num:num + opts['zdim']])
                    tmp = tf.reduce_sum(tmp, axis=0)
                    related_samples.append(tmp[np.newaxis, :])
            else:
                random_num=tf.get_variable('schimit_ph%d'%(classes),[opts['zdim'],j],
                                         initializer=tf.random_uniform_initializer(-1.,1.),trainable=True)
                for i in range(opts['zdim']):
                    tmp = random_num[i, :]
                    tmp = tf.tile(tmp[:, np.newaxis], [1, opts['zdim']])
                    tmp =  tf.multiply(tmp,encode_schimit[num:num+j])
                    tmp = tf.reduce_sum(tmp,axis=0)
                    related_samples.append(tmp[np.newaxis,:])
        related_samples = tf.concat(related_samples,axis=0)
    return related_samples

def matmul_mulelms(*matrixs):
    '''
    连乘函数。将输入的矩阵按照输入顺序进行连乘。

    Parameters
    ----------
    *matrixs : 矩阵
        按计算顺序输入参数.

    Raises
    ------
    ValueError
        当参数个数小于2时，不满足乘法的要求.

    Returns
    -------
    res : 矩阵
        返回连乘的结果.

    '''
    if len(matrixs)<2:
        raise ValueError('Please input more than one parameters.')
    res = matrixs[0]
    for i in range(1,len(matrixs)):
        res = np.matmul(res, matrixs[i])
    return res

# 3.3.4 施密特正交化
def One_Col_Matrix(array):
    '''
    确保为列矩阵

    Parameters
    ----------
    array : 矩阵，向量或数组

    Raises
    ------
    ValueError
        获得的参数不是1xn或mx1时，报错.

    Returns
    -------
    TYPE
        返回列矩阵.

    '''
    mat = np.mat(array)
    if mat.shape[0] == 1:
        return mat.T
    elif mat.shape[1] == 1:
        return mat
    else:
        raise ValueError('Please input 1 row array or 1 column array')

def Transfor_Unit_Vector(matrix):
    '''
    将每列都转换为标准列向量，即模等于1

    Parameters
    ----------
    matrix : 矩阵

    Returns
    -------
    unit_mat : 矩阵
        每列模都为1的矩阵.

    '''
    col_num = matrix.shape[1]
    # 初始化为零矩阵
    unit_mat = np.zeros((matrix.shape))
    for col in range(col_num):
        vector = matrix[:,col]
        unit_vector = vector / np.linalg.norm(vector)
        unit_mat[:,col] = unit_vector.T
    return unit_mat

def Gram_Schmidt_Orthogonality(matrix):
    '''
    施密特正交化方法

    Parameters
    ----------
    matrix : 矩阵

    Returns
    -------
    标准正交化矩阵。

    '''
    col_num = matrix.shape[1]
    # 第一列无需变换
    gram_schmidt_mat = One_Col_Matrix(matrix[:,0])
    for col in range(1,col_num):
        raw_vector = One_Col_Matrix(matrix[:,col])
        orthogonal_vector = One_Col_Matrix(matrix[:,col])
        if len(gram_schmidt_mat.shape)==1:
            # 当矩阵为列向量是，shape的返回值为“(row,)”，没有col的值
            gram_schmidt_mat_col_num = 1
        else:
            gram_schmidt_mat_col_num = gram_schmidt_mat.shape[1]
        for base_vector_col in range(gram_schmidt_mat_col_num):
            base_vector = gram_schmidt_mat[:,base_vector_col]
            prejective_vector = matmul_mulelms(base_vector, linalg.inv(np.matmul(base_vector.T,base_vector)), base_vector.T, raw_vector)
            orthogonal_vector = orthogonal_vector - prejective_vector
        gram_schmidt_mat = np.hstack((gram_schmidt_mat,orthogonal_vector))
    #print(gram_schmidt_mat)
    return Transfor_Unit_Vector(gram_schmidt_mat)
    # return gram_schmidt_mat

def repeat_elements(x, rep, axis):
    x_shape = x.get_shape().as_list()
    # For static axis
    x_rep=[]
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
        # repeat each slice the given number of reps
        for classnum,classsample in enumerate(splits):
            x_rep.append(tf.tile(classsample,[rep[classnum],1]))
        # x_rep = [s for s in splits for i in range(rep[i])]
    return tf.concat(x_rep, axis)

def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return tf.reduce_mean(loss)

def crossentropy(logits, targets, weight=None, reduction='none'):
    """N samples, C classes
    logits: [N, C]
    targets:[N] range [0, C-1]
    weight: [C]
    """
    # C = tf.shape(logits)[1]
    # if weight is not None:
    #     assert len(weight)==C, 'weight length must be equal to classes number'
    #     assert weight.dim() == 1, 'weight dim must be 1'
    # else:
    #     weight = tf.ones(C)
    prob = tf.nn.softmax(logits, dim=1)
    log_prob = tf.log(prob+1e-07)
    tar_one_hot = tf.one_hot(targets,depth=9 ,dtype=tf.float32)#.type(tf.float32)
    targets = tf.cast(targets, dtype = tf.int64)

    loss = tf.gather(weight,targets)
    tmp = (-log_prob * tar_one_hot)
    tmp = tf.reduce_sum(tmp,reduction_indices=1)
    loss = loss * tmp
    if reduction == 'mean':
        temp = tf.gather(weight,targets)
        temp = tf.reduce_sum(temp)
        loss_sum = tf.reduce_sum(loss)
        loss_result = loss_sum / (temp + 1e-7)
    elif reduction == 'none':
        loss_result = loss
    return loss_result

def IBLoss(input, target, features,alpha,epsilon,weight):
    #class_num = opts['n_classes']
    grads = tf.reduce_sum(tf.abs(tf.nn.softmax(input, dim=1) - tf.one_hot(target, 9)),1) # N * 1
    features = tf.reshape(features, [-1])
    ib = grads*features
    ib = alpha / (ib + epsilon)
    loss_result = crossentropy(input, target, weight=weight, reduction='none')
    #return ib
    return ib_loss(loss_result, ib)

# def crossentropy(logits, targets, weight=None, reduction='none'):
#     """N samples, C classes
#     logits: [N, C]
#     targets:[N] range [0, C-1]
#     weight: [C]
#     """
#     prob = tf.nn.softmax(logits, dim=1)
#     loss = -weight * (targets * tf.log(prob+1e-07) + (1 - targets) * (tf.log(1 - prob + 1e-07)))
#     loss = tf.reduce_sum(loss) / tf.to_float(tf.size(targets))
#     return loss
# def CB_loss(opts, labels, logits, beta):
#     samples_per_cls = opts['dataset_ratio_mapping']
#     no_of_classes = opts['n_classes']
#     effective_num = 1.0 - np.power(beta, samples_per_cls)
#     weights = (1.0 - beta) / np.array(effective_num)
#     weights = weights / np.sum(weights) * no_of_classes

#     labels_one_hot = tf.one_hot(indices=labels, depth=no_of_classes,dtype=np.float32)

#     weights = tf.to_float(tf.convert_to_tensor(weights))
#     weights = tf.expand_dims(weights,0)
#     weights = tf.tile(weights,[tf.shape(logits)[0],1])
#     weights = weights* labels_one_hot
#     weights = tf.reduce_sum(weights,1)
#     weights = tf.expand_dims(weights,1)
#     weights = tf.tile(weights,[1,no_of_classes])

#     cb_loss = crossentropy(logits = logits,targets = labels_one_hot, weight = weights)
    
#     return cb_loss