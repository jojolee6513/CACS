import tensorflow as tf
import os
# import ops
import numpy as np
from datahandler import datashapes
from tensorflow.python.ops import gen_nn_ops
from ops import linear,conv3d,batch_norm,deconv3d,reparameter1,schimit

def encoder(opts, inputs, kernel, reuse=tf.AUTO_REUSE, is_training=False):
    with tf.variable_scope("encoder", reuse=reuse):
        return dcgan_encoder(opts, inputs, kernel, is_training, reuse)

def classifier1(opts, noise, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("classifier", reuse=reuse):
        if opts['mlp_classifier']:
            out = linear(opts, noise, 500, 'mlp1')
            out = tf.nn.relu(out)
            out = linear(opts, out, 500, 'mlp2')
            out = tf.nn.relu(out)
            logits = linear(opts, out, opts['n_classes'], 'classifier')
        else:
            logits = linear(opts, noise, opts['n_classes'], 'classifier')
            #logits = tf.nn.softmax(logits)
    return logits


def decoder(opts, noise, layerx, window, kernel, reuse=tf.AUTO_REUSE, is_training=True):
    with tf.variable_scope("generator", reuse=reuse):
        res = dcgan_decoder(opts, noise,layerx, window, kernel, is_training, reuse)
        return res

def dcgan_encoder(opts, inputs, kernel, is_training=False, reuse=False):
    layer_x = inputs
    #for e in y:
    layer_x = conv3d(opts, layer_x, kernel,d_d=2, d_h=1, d_w=1, conv_filters_dim=[1, 1, 7], padding='VALID', scope='h0_conv')
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training,reuse, scope='h0_bn')
    layer_x = tf.nn.relu(layer_x)
    layer_x_first = layer_x
    layer_x = conv3d(opts, layer_x, kernel, d_d=1, d_h=1, d_w=1, conv_filters_dim=[1,1,7],padding='SAME',scope='h1_conv')##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training,reuse, scope='h1_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x = conv3d(opts, layer_x, kernel,  d_d=1, d_h=1, d_w=1, conv_filters_dim=[1, 1, 7], padding='SAME',
                         scope='h2_conv')  ##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h2_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x += layer_x_first

    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h3_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x = conv3d(opts, layer_x, 128, conv_filters_dim=[1, 1, 49], padding='VALID', scope='h3_conv')
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training,reuse, scope='h4_bn')
    layer_x = tf.nn.relu(layer_x)
    layer_x_shape = layer_x.get_shape().as_list()
    layer_x = tf.reshape(layer_x,(-1, layer_x_shape[1],layer_x_shape[2], layer_x_shape[4],layer_x_shape[3]))

    layer_x = conv3d(opts, layer_x, kernel, d_d=1, d_h=1, d_w=1, conv_filters_dim=[3, 3, 128], padding='VALID',
                         scope='h4_conv')  ##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training,reuse, scope='h5_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x_first = layer_x
    layer_x = conv3d(opts, layer_x, kernel, d_d=1, d_h=1, d_w=1, conv_filters_dim=[3,3,1],padding='SAME',scope='h5_conv')##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training,reuse, scope='h6_bn')
    layer_x = tf.nn.relu(layer_x)
    layer_x = conv3d(opts, layer_x, kernel, d_d=1, d_h=1, d_w=1, conv_filters_dim=[3, 3, 1], padding='SAME',
                         scope='h6_conv')  ##
    layer_x_first = conv3d(opts, layer_x_first, kernel, d_d=2, d_h=1, d_w=1, conv_filters_dim=[1, 1, 1],
                               padding='VALID',scope='h7_conv')  ##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h7_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x += layer_x_first

    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h8_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x_shape = layer_x.get_shape().as_list()
    layer_x_pool = tf.nn.pool(layer_x, window_shape = [layer_x_shape[1],layer_x_shape[2],layer_x_shape[3]], pooling_type='AVG',padding='VALID')
    layer_x_2d = tf.squeeze(layer_x_pool,[1,2,3])
    feature = tf.reduce_sum(tf.abs(layer_x_2d), axis=1)
    feature = tf.reshape(feature,[-1,1])
    #d2 = tf.layers.dropout(layer_x_2d, rate=0.5, training=is_training)
    z = linear(opts, layer_x_2d, opts['zdim'], scope='z_ph')
    return z,layer_x,feature

def dcgan_decoder(opts, noise,layer_x, window, kernel, is_training=False, reuse=False):
    h0 = linear(opts, noise, 24, scope='h0_lin')
    h0 = tf.expand_dims(h0,1)
    h0 = tf.expand_dims(h0,1)
    batch_size = tf.shape(noise)[0]
    #layer_x = tf.expand_dims(layer_x_3d,3)
    layer_x_first = layer_x
    layer_x = deconv3d(opts, layer_x, [batch_size, window-2,window-2,1, kernel], d_d=1, d_h=1, d_w=1, conv_filters_dim=[3, 3, 1], padding='SAME',
                         scope='h0_deconv')  ##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h0_bn')
    layer_x = tf.nn.relu(layer_x)
    layer_x = deconv3d(opts, layer_x, [batch_size,window-2,window-2, 1, kernel], d_d=1, d_h=1, d_w=1, conv_filters_dim=[3, 3, 1], padding='SAME',
                         scope='h1_deconv')  ##
    layer_x_first = deconv3d(opts, layer_x_first, [batch_size, window-2,window-2,1, kernel], d_d=2, d_h=1, d_w=1, conv_filters_dim=[1, 1, 1],
                               padding='VALID', scope='h2_deconv')  ##
    layer_x += layer_x_first
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h1_bn')
    layer_x = tf.nn.relu(layer_x)
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h2_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x = deconv3d(opts, layer_x, [batch_size,window,window,128,1], d_d=1, d_h=1, d_w=1, conv_filters_dim=[3, 3, 128], padding='VALID',
                         scope='h3_deconv')  ##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h3_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x_shape = layer_x.get_shape().as_list()
    layer_x = tf.reshape(layer_x,(-1, layer_x_shape[1],layer_x_shape[2], layer_x_shape[4],layer_x_shape[3]))
    layer_x = deconv3d(opts, layer_x, [batch_size,window,window,49,kernel], conv_filters_dim=[1, 1, 49], padding='VALID', scope='h4_deconv')
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training,reuse, scope='h4_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x_first = layer_x
    layer_x = deconv3d(opts, layer_x, [batch_size,window,window,49,kernel], d_d=1, d_h=1, d_w=1, conv_filters_dim=[1, 1, 7], padding='SAME',
                         scope='h5_deconv')  ##
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h5_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x = deconv3d(opts, layer_x, [batch_size,window,window,49,kernel], d_d=1, d_h=1, d_w=1, conv_filters_dim=[1, 1, 7], padding='SAME',
                         scope='h6_deconv')  ##
    layer_x += layer_x_first
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h6_bn')
    layer_x = tf.nn.relu(layer_x)
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training, reuse, scope='h7_bn')
    layer_x = tf.nn.relu(layer_x)

    layer_x = deconv3d(opts, layer_x, [batch_size,window,window,103,1],d_d=2, d_h=1, d_w=1, conv_filters_dim=[1, 1, 7], padding='VALID', scope='h7_deconv')
    if opts['batch_norm']:
        layer_x = batch_norm(opts, layer_x, is_training,reuse, scope='h8_bn')
    layer_x = tf.nn.relu(layer_x)
    return layer_x

# def dcgan_decoder(opts, noise, is_training=False, reuse=False):
    # output_shape = datashapes[opts['dataset']]
    # num_units = opts['g_num_filters']
    # batch_size = tf.shape(noise)[0]
    # num_layers = opts['g_num_layers']
    # height = 7
    # width = 7
    # depth = 18
    # h0 = ops.linear(opts, noise,  height * width* depth* num_units, scope='h0_lin')
    # h0 = tf.reshape(h0, [-1, height, width, depth, num_units])
    # h0 = tf.nn.relu(h0)
    # layer_x = h0
    # for i in range(num_layers -1):
    #     scale = (i + 1)
    #     _out_shape = [batch_size, height +2 * scale,
    #                   width + 2* scale, depth+ 4*scale-2, num_units // (scale * 2)]
    #     layer_x = ops.deconv3d(opts, layer_x, _out_shape, conv_filters_dim=[3,3,3+i*2],
    #                            scope='h%d_deconv' % i,  padding='VALID')
    #     if opts['batch_norm']:
    #         layer_x = ops.batch_norm(opts, layer_x,
    #                                  is_training, reuse, scope='h%d_bn' % i)
    #     layer_x = tf.nn.relu(layer_x)
    # _out_shape = [batch_size] + list(output_shape)
    # last_h = ops.deconv3d(
    #     opts, layer_x, _out_shape, d_h=1, d_w=1, d_d =1,scope='hfinal_deconv', conv_filters_dim=[3,3,7], padding='VALID')
    # return tf.nn.sigmoid(last_h)

def reparameter(opts, augrate,cls_cnt, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("reparameter",reuse=reuse):
        gan_z = np.random.uniform(-1.0, 1.0, [opts['n_classes'], opts['zdim']])
        # batchsize = input.get_shape().as_list()
        # batchsize = int(batchsize[0])
        z1,z2 = reparameter1(opts, gan_z, cls_cnt,augrate, "reparameter")
        #batchsize = input.get_shape().as_list()
        return z1,z2

def schimit_Orthogonalization(opts, schimit_ji, cls_cnt, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("reparameter_schmitd", reuse=reuse):
        return schimit(opts, schimit_ji, "reparameter_schmitd",reuse,cls_cnt)