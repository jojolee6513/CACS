import logging
import math
import os
import time
from utils import shuffle
import numpy as np
import tensorflow as tf
from sympy import *
from tqdm import tqdm
import scipy.io as sio
import utils
from datahandler import datashapes
from models import encoder, decoder, classifier1,reparameter,schimit_Orthogonalization
from ops import IBLoss,Gram_Schmidt_Orthogonality,augment_batch

class DGC(object):

    def __init__(self, opts, tag, alpha,window,kernel,augrate,cls_cnt):
        tf.reset_default_graph()
        logging.error('Building the Tensorflow Graph')
        gpu_options = tf.GPUOptions(allow_growth=True,)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.opts = opts
        self.alpha = alpha
        self.epsilon = 0.01
        self.window = window
        self.kernel = kernel
        self.augrate = augrate
        self.cls_cnt = cls_cnt

        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        # Add placeholders
        shape = [window,window,opts['num_pcs'],1]
        self.sample_points = tf.placeholder(
            tf.float32, [None] + shape, name='real_points_ph')
        self.labels = tf.placeholder(tf.int64, shape=[None], name='label_ph')

        # self.sample_noise = tf.placeholder(
        #     tf.float32, [None, opts['zdim']], name='noise_ph')
        # self.repeatnum = tf.placeholder(tf.int64, shape=[None])
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.encodedz = tf.placeholder(tf.float32, shape=[None, opts['zdim']])
        self.z_aug = tf.placeholder(tf.float32, shape=[None, opts['zdim']])
        #self.z_true = tf.placeholder(tf.float32, [None, opts['zdim']])
        self.y_aug = tf.placeholder(tf.int64, shape=[None])
        # self.sample_noise_repeat1 = tf.placeholder(tf.float32, shape=[None, opts['zdim']], name='sample_noise_repeat')
        self.encode_repeat = tf.placeholder(tf.float32, shape=[None, opts['zdim']], name='sample_noise_repeat')
        self.encode_schimit = tf.placeholder(tf.float32, shape=[299, opts['zdim']], name='encode_schimit')
        # self.sample_label = tf.placeholder(tf.int64, shape=[None])
        self.cls_data = tf.placeholder(tf.float32, shape=[None,opts['zdim']])
        self.weight = tf.placeholder(tf.float32, shape=[opts['n_classes']])
        # self.generate_samples = tf.placeholder(tf.float32, shape = [opts['zdim']*opts['n_classes'],opts['zdim']])
        # self.generate_labels = tf.placeholder(tf.int64, shape = [opts['zdim']*opts['n_classes']])
        #self.z2_grad = tf.placeholder(tf.float32, shape=[opts['zdim']*opts['n_classes'], opts['zdim']])
        #self.new_schimit_aug_grad = tf.placeholder(tf.float32, shape=[opts['zdim']*opts['n_classes'], opts['zdim']])
        # Build training computation graph
        sample_size = tf.shape(self.sample_points)[0]
        self.encoded,self.layerx,self.feature = encoder(opts, inputs=self.sample_points, kernel=self.kernel, is_training=self.is_training)
        self.z1,self.z2 = reparameter(opts, self.augrate, cls_cnt=self.cls_cnt,reuse=tf.AUTO_REUSE)

        #self.encoded = self.get_encoded(opts, self.enc_mean, self.enc_sigmas) #z
        #self.encoded2 = self.batch_z#augmented_z
        #self.dimension_transform = dimension_transform(opts, self.sample_points, sample_size, reuse=tf.AUTO_REUSE)
        self.reconstructed1 = decoder(opts, noise=self.encoded,layerx=self.layerx, window=self.window, kernel=self.kernel, is_training=self.is_training) #x_hat, theta_1; self.encoded=(?,64)

        #self.reconstructed2 = decoder(opts, noise=self.sample_noise_repeat, is_training=self.is_training)

        self.probs1 = classifier1(opts, self.encoded) #theta_2


        self.correct_sum = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.probs1, axis=1), self.labels), tf.float32))
        #self.decoded = decoder(opts, noise=self.sample_noise, is_training=self.is_training) #self.sample_noise=(?,64)
        self.new_schimit = schimit_Orthogonalization(opts, self.encode_schimit,cls_cnt=self.cls_cnt, reuse=tf.AUTO_REUSE)
        self.generate_samples,self.generate_labels = augment_batch(opts, self.cls_data, self.new_schimit,self.augrate,self.cls_cnt)
        self.probs2 = classifier1(opts, self.generate_samples)  # theta_2
        #self.reconstructed2 = decoder(opts, noise=self.generate_samples, is_training=self.is_training)
        self.loss_cls = self.cls_loss(self.labels, self.probs1)
        self.loss_cls1 = self.cls_loss(self.generate_labels, self.probs2)
        self.loss_cls2 = self.cls_loss2(self.labels, self.probs1,self.feature,self.alpha, self.epsilon, weight = self.weight)
        # self.loss_cls3 = self.cls_loss(self.generate_labels, self.probs2)
        #self.loss_cls3 = self.cls_loss(self.tr_fa ,self.tr_fa_ph)
        #self.loss_mmd1 = self.mmd_penalty(self.z_aug, self.sample_noise_repeat)
        self.loss_mmd1 = self.mmd_penalty(self.cls_data, self.z1)
        self.loss_mmd = self.mmd_penalty(self.z2, self.generate_samples)
        #self.loss_mmd2 = self.mmd_penalty(self.z2_grad, self.new_schimit_aug_grad)
        self.loss_recon = self.reconstruction_loss(self.opts, self.sample_points, self.reconstructed1)
        #self.loss_recon2 = self.reconstruction_loss(self.opts, self.sample_points, self.reconstructed2)+ self.loss_recon2
        self.objective = self.loss_recon + self.loss_cls # + opts['lambda'] * self.loss_mmd1 ##+ self.loss_cls1
        self.objective1 = self.loss_recon + self.loss_cls2 #+ self.loss_cls3 + opts['lambda'] * self.loss_mmd
        self.objective2 = self.loss_mmd+self.loss_mmd1+self.loss_cls
        # Build evaluate computation graph
        logpxy = []
        dimY = opts['n_classes']
        N = sample_size
        S = opts['sampling_size']
        x_rep = tf.tile(self.sample_points, [S, 1, 1, 1, 1])
        for i in range(dimY):
            y = tf.fill((N,), i)
            z,layer_x,feature = encoder(opts, inputs=self.sample_points, kernel=self.kernel, is_training=False)
            z = tf.tile(z, [S, 1])
            layer_x = tf.tile(layer_x, [S,1, 1, 1, 1])
            # mu = tf.tile(mu, [S, 1])
            # log_sig = tf.tile(log_sig, [S, 1])
            y = tf.tile(y, [S, ])
            # z = self.get_encoded(opts, mu, log_sig)
            z_sample = tf.random_normal((tf.shape(z)[0], opts['zdim']), 0., 1., dtype=tf.float32)

            mu_x = decoder(opts, noise=z,layerx=layer_x,window=self.window, kernel=self.kernel, is_training=False)
            logit_y = classifier1(opts, z)
            logp = -tf.reduce_sum((x_rep - mu_x) ** 2, axis=[1, 2, 3, 4])
            log_pyz = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit_y)
            mmd_loss = self.mmd_penalty(z_sample, z)
            bound = 0.5 * logp + log_pyz + mmd_loss #opts['lambda'] *
            bound = tf.reshape(bound, [S, N])
            bound = self.logsumexp(bound) - tf.log(float(S))
            logpxy.append(tf.expand_dims(bound, 1))
        logpxy = tf.concat(logpxy, 1)
        y_pred = tf.nn.softmax(logpxy)
        self.eval_probs = y_pred

        self.loss_pretrain = self.pretrain_loss() if opts['e_pretrain'] else None
        self.add_optimizers()
        self.add_savers()
        self.tag = tag

    def log_gaussian_prob(self, x, mu=0.0, log_sig=0.0):
        logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                  - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
        ind = list(range(1, len(x.get_shape().as_list())))
        return tf.reduce_sum(logprob, ind)

    def logsumexp(self, x):
        x_max = tf.reduce_max(x, 0)
        x_ = x - x_max
        tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-20, np.inf))
        return tmp + x_max

    def add_savers(self):
        saver = tf.train.Saver(max_to_keep=11)
        tf.add_to_collection('real_points_ph', self.sample_points)
        #tf.add_to_collection('gan_z_ph', self.gan_z)
        tf.add_to_collection('is_training_ph', self.is_training)
        # if self.enc_mean is not None:
        #     tf.add_to_collection('encoder_mean', self.enc_mean)
        #     tf.add_to_collection('encoder_var', self.enc_sigmas)
        tf.add_to_collection('encoder', self.encoded)
        #tf.add_to_collection('decoder', self.decoded)
        # tf.add_to_collection('reparameter', self.sample_noise_repeat)
        tf.add_to_collection('schmidt', self.new_schimit)
        self.saver = saver

    def cls_loss(self, labels, logits):
        return tf.reduce_mean(tf.reduce_sum(  # FIXME
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)))
    def cls_loss2(self, labels, logits,feature,alpha, epsilon, weight):
        return IBLoss(logits, labels, feature, alpha, epsilon, weight)
        # return CB_loss(self.opts, labels, logits,  beta=alpha)

    def mmd_penalty(self, sample_pz, sample_qz):
        opts = self.opts
        sigma2_p = 1.
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        Cbase = 2. * opts['zdim'] * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat

    def reconstruction_loss(self, opts, real, reconstr):
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.5 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        return tf.train.AdamOptimizer(lr)

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        #classifier_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier2')
        reparameter_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reparameter')
        reparameter_schmitd_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reparameter_schmitd')
        #dimension_transform_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dimension_transform') + dimension_transform_vars
        ae_vars = encoder_vars + decoder_vars + classifier_vars #+reparameter_vars#+reparameter_vars0+reparameter_vars1+reparameter_vars2+reparameter_vars3+reparameter_vars4+reparameter_vars5+reparameter_vars6+reparameter_vars7+reparameter_vars8

        # Auto-encoder optimizer
        opt = self.optimizer(lr, self.lr_decay)
        self.ae_opt = opt.minimize(loss=self.objective,
                                   var_list=ae_vars)
        self.ae_opt1 = opt.minimize(loss=self.objective1,
                                   var_list=ae_vars)
        # self.cls_opt = opt.minimize(loss=self.loss_cls1,
        #                             var_list=classifier_vars)
        # self.cls_opt2 = opt.minimize(loss=self.loss_cls2, #+self.loss_cls3,
        #                              var_list=classifier_vars + classifier_vars2 )+reparameter_vars+reparameter_schmitd_vars
        self.reparameter_opt = opt.minimize(loss=self.loss_mmd + self.loss_cls1 + self.loss_mmd1, #+self.loss_cls1, classifier_vars+
                                            var_list=reparameter_schmitd_vars+reparameter_vars + classifier_vars)
        # Encoder optimizer
        if opts['e_pretrain']:
            opt = self.optimizer(lr)
            # self.pretrain_opt = opt.minimize(loss=self.loss_pretrain,
            #                                  var_list=encoder_vars)
        else:
            self.pretrain_opt = None
        if opts['LVO']:
            self.lvo_opt = opt.minimize(loss=self.objective, var_list=encoder_vars)

    def sample_pz(self, num=100, z_dist=None, labels=None):
        opts = self.opts
        if z_dist is None:
            mean = np.zeros(opts["zdim"])
            cov = np.identity(opts["zdim"])
            noise = np.random.multivariate_normal(mean, cov, num).astype(np.float32)
            return noise
        assert labels is not None
        a = np.identity(z_dist)
        a = a[np.newaxis, :]
        a = (np.zeros((9,z_dist)), (a.repeat(9,axis=0)))
        means, covariances = a
        noise = np.array([np.random.multivariate_normal(means[e], covariances[e]) for e in labels])
        return noise

    def augment_z(self, x, class_cnt):
        n_classes = len(class_cnt)
        x_aug_list = []
        for i in range(n_classes):
            xi = x[i,:]
            xi = xi[np.newaxis,:]
            x_aug_list.append(np.repeat(xi,repeats=class_cnt[i],axis=0))
        if len(x_aug_list[0]) == 0:
            return x
        aug = np.concatenate(x_aug_list, axis=0)
        return aug
    
    def get_lr_decay(self, opts, epoch):
        decay = 1.
        if opts['lr_schedule'] == "manual":
            if epoch == 30:
                decay = decay / 10.
            if epoch == 50:
                decay = decay / 100.
            if epoch == 100:
                decay = decay / 150.
            if epoch == 150:
                decay = decay / 100.
            # if epoch == 200:
            #     decay = decay / 100.
        elif opts['lr_schedule'] == "manual_smooth":
            enum = opts['epoch_num']
            decay_t = np.exp(np.log(100.) / enum)
            decay = decay / decay_t
        return decay

    def train(self, data,seed,star_epoch):
        opts = self.opts
        if opts['verbose']:
            logging.error(opts)
        losses = []
        losses_rec = []
        losses_match = []
        losses_cls1 = []
        losses_cls2 = []
        data_batch = []
        data_batch_label = []
        # cls_data = []
        # cls_label = []
        batches_num = math.ceil(data.num_points / opts['batch_size'])
        few_class = [2, 4, 6, 8,0,3,5,7]
        for i, j in enumerate(few_class):
            idx = (data.labels == j)
            n_c = np.count_nonzero(idx)-1
            tmp_few = data.data[idx]
            label_few = data.labels[idx]
            if i==0:
                for k in range(batches_num):
                    data_batch.append(tmp_few[n_c, np.newaxis, :, :, :, :])
                    data_batch_label.append([label_few[n_c]])
                    n_c=n_c-1
            while(n_c+1):
                for k in range(batches_num):
                    if(n_c+1):
                        data_batch[k] = (np.concatenate((data_batch[k], tmp_few[n_c,np.newaxis, :, :, :, :]),axis=0))
                        data_batch_label[k] = np.concatenate((data_batch_label[k], [label_few[n_c]]),axis=0)
                        n_c=n_c-1
        # many_class = [0, 1, 3, 5, 7]
        idx =  (data.labels == 1) #| (data.labels == 3) | (data.labels == 5) | (data.labels == 7) (data.labels == 0) |
        tmp = data.data[idx]
        tmp_label = data.labels[idx]
        pre_few_class_cnt,new_class_cut = 0,0
        for k in range(batches_num):
            if k:
                pre_few_class_cnt = new_class_cut
            new_class_cut += data_batch[k].shape[0]
            data_batch[k] = np.concatenate((data_batch[k],
                            tmp[k*opts['batch_size']-pre_few_class_cnt:(k+1)*opts['batch_size']-new_class_cut, :, :, :, :]))
            data_batch_label[k] = np.concatenate((data_batch_label[k],
                                tmp_label[k*opts['batch_size']-pre_few_class_cnt:(k+1)*opts['batch_size']-new_class_cut]))
            np.random.seed(seed)
            np.random.shuffle(data_batch[k])
            np.random.seed(seed)
            np.random.shuffle(data_batch_label[k])
        self.sess.run(tf.global_variables_initializer())###

        counter = 0
        start_time = time.time()
        for epoch in range(opts["epoch_num"]):
            # Update learning rate if necessary
            
            decay = self.get_lr_decay(opts, epoch)
            # # Save the model
            # if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
            #     self.saver.save(self.sess,
            #                     os.path.join(opts['work_dir'], 'checkpoints', 'trained'),
            #                     global_step=counter)
            acc_total = 0.
            loss_total = 0.
            loss_cls_batch=0.
            loss_rec_batch=0.
            loss_match1_batch=0.
            encodedz_new = []
            batch_labels_new = []

            for it in tqdm(range(batches_num)):
                batch_images = data_batch[it]
                batch_labels = data_batch_label[it]
                train_size = batch_labels.shape
                class_cnt = [np.count_nonzero(batch_labels == n) for n in range(opts['n_classes'])]
                if epoch<star_epoch:
                    feed_dict1 = {
                        self.sample_points: batch_images,
                        self.labels: batch_labels,
                        self.lr_decay: decay,
                        self.is_training: True}
                    # z_aug, y_aug = self.augment_batch(encoded, batch_labels, gan_z, class_cnt)
                    (_, encodedz, loss, loss_rec, loss_cls, correct) = self.sess.run(
                        [self.ae_opt, self.encoded,self.objective,
                         self.loss_recon, self.loss_cls, #self.loss_cls1, loss_cls1,
                         self.correct_sum],
                        feed_dict=feed_dict1)
                else:
                    per_cls_weights = 1.0 / np.array(class_cnt)
                    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_cnt)
                    feed_dict1 = {
                        self.sample_points: batch_images,
                        self.labels: batch_labels,
                        self.weight: per_cls_weights,
                        self.lr_decay: decay,
                        self.is_training: True}
                    # z_aug, y_aug = self.augment_batch(encoded, batch_labels, gan_z, class_cnt)
                    (_, encodedz, loss, loss_rec, loss_cls, correct) = self.sess.run(
                        [self.ae_opt1, self.encoded,self.objective1,
                         self.loss_recon, self.loss_cls2, self.correct_sum],
                        feed_dict=feed_dict1)
                encodedz_new.append(encodedz)
                batch_labels_new.append(batch_labels)
                acc_total += correct / train_size[0]
                loss_total += loss
                loss_rec_batch+=loss_rec
                loss_cls_batch+=loss_cls
                counter += 1
                z_printline = []
                for i in range(opts['n_classes']):
                    idx = (batch_labels == i)
                    z_printline.append(encodedz[idx])

            cls_data = []
            encode_schimit = []
            encodedz_new = np.concatenate(encodedz_new,axis=0)
            batch_labels_new = np.concatenate(batch_labels_new,axis=0)
            for i in range(opts['n_classes']):
                idx = (batch_labels_new == i)
                n_c = np.count_nonzero(idx)
                cls_data.append(encodedz_new[idx])
                encode_tmp = cls_data[i][0:opts['zdim']] # encode_tmp = utils.Linearly_independent(cls_data[i])
                encode_schimit.append(np.transpose(Gram_Schmidt_Orthogonality(np.transpose(encode_tmp))))    
            encode_schimit = np.concatenate(encode_schimit, axis=0)
            cls_data = np.concatenate(cls_data, axis=0)
            # for i in range(opts['n_classes']):
            #     z_aug.append(np.repeat(z[i][np.newaxis, :], opts['zdim'], axis=0))
            # z_aug = np.concatenate(z_aug,axis=0)
            # (new_schimit_aug) = self.sess.run(  #, loss_match, loss_cls1
            #     [self.reparameter_opt,self.z2,self.new_schimit],  ##, self.loss_mmd,self.loss_cls1
            #     feed_dict={
            #         self.encode_schimit: encode_schimit,
            #         self.lr_decay: decay,
            #         self.is_training: True
            #     })
            # new_schimit_aug_grad = np.zeros((576,64))
            # z2_grad = np.zeros((576,64))
            # for i in range(64):
            #     if i ==0:
            #         new_schimit_aug_grad[:,0] = new_schimit_aug[:,0]
            #         z2_grad[:,0] = z2[:,0]
            #     else:
            #         new_schimit_aug_grad[:,i] = new_schimit_aug[:,i]-new_schimit_aug[:,i-1]
            #         z2_grad[:,i] = z2[:,i]-z2[:,i-1]
            # generate_samples,generate_labels= self.augment_batch(opts, cls_data, new_schimit_aug,self.augrate)
            (_, loss_match, loss_cls1,loss_match1) = self.sess.run(  ##loss_match2,
                [self.reparameter_opt,self.loss_mmd,self.loss_cls1,self.loss_mmd1],  ##self.loss_mmd2,
                feed_dict={
                    #self.z2_grad:z2_grad,
                    #self.new_schimit_aug_grad:new_schimit_aug_grad,
                    # self.generate_samples:generate_samples,
                    # self.generate_labels:generate_labels,
                    self.cls_data:cls_data,
                    self.encode_repeat: encodedz_new,
                    self.encode_schimit: encode_schimit,
                    self.lr_decay: decay,
                    self.is_training: True})
            # sio.savemat('encode.mat',
            #             mdict={'encode': cls_data, 'schimidt': new_schimit_aug, 'cls_encode_z': z2})
            loss_match1_batch+=loss_match+loss_match1
            loss_cls_batch+=loss_cls1
            loss_total+=loss_match +loss_cls1+loss_match1
            losses.append(loss_total)
            losses_rec.append(loss_rec_batch)
            losses_match.append(loss_match1_batch)
            losses_cls1.append(loss_cls_batch)
            #losses_cls2.append(loss_cls1)
            # loss = loss + loss_cls1
            #new_schimit_aug = self.pretrain_encoder(data, encode_new, data_batch_label)
            # if epoch%30==0 and epoch<90
            # Print debug info
            now = time.time()
            debug_str = 'EPOCH: %d/%d, BATCH/SEC:%.2f' \
                        % (epoch + 1, opts['epoch_num'], float(counter) / (now - start_time))
            debug_str += ' (TOTAL_LOSS=%.5f, RECON_LOSS=%.5f, MATCH_LOSS=%.5f, CLS_LOSS1=%.5f)' % (
                losses[-1], losses_rec[-1], losses_match[-1], losses_cls1[-1]) #, losses_cls2[-1] , CLS_LOSS2=%.5f
            logging.error(debug_str)
            training_acc = acc_total / batches_num
            avg_loss = loss_total / batches_num
            print("Train loss: %.5f, Train acc: %.5f, Time: %.5f" % (avg_loss, training_acc, time.time() - start_time))

            if epoch > 0 and epoch % 10 == 0:
                self.saver.save(self.sess,
                                os.path.join(opts['work_dir'],
                                             'checkpoints',
                                             'trained-final'),
                                global_step=epoch)
                para_num = sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
                # para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
                para_size = para_num * 4 / 1024 / 1024
                graph =tf.get_default_graph()
                flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
                params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
                print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

                # checkpoint = tf.train.Checkpoint(model = self.sess)
                # checkpoint.save(os.path.join(opts['work_dir'],'checkpoints','checkpoints'))
            if ((epoch+1)%150==0): # (epoch+1==star_epoch) or
                now_time=time.time()
                print("train_time: %.5f" % (now_time - start_time))
                
                self.evaluate(data, epoch,star_epoch)
                print("test_time: %.5f" % (time.time() - start_time))

    def evaluate(self, data, epoch,star_epoch):
        batch_size = self.opts['batch_size']
        batches_num = math.ceil(len(data.test_data) / batch_size)
        probs = []
        start_time = time.time()
        for it in tqdm(range(batches_num)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size
            prob = self.sess.run(
                self.eval_probs,
                feed_dict={self.sample_points: data.test_data[start_idx:end_idx],
                           self.is_training: False})
            probs.append(prob)
        probs = np.concatenate(probs, axis=0)
        predicts = np.argmax(probs, axis=-1)
        asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi = utils.get_test_metrics(data.test_labels, predicts)
        print("EPOCH: %d, ASCA=%.5f, PRE=%.5f, REC=%.5f, SPE=%.5f, F1_ma=%.5f, F1_mi=%.5f, G_ma=%.5f, G_mi=%.5f" % (
            epoch, asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi))
        each_acc, average_acc, overall_acc, kappa, precision, average_precision = utils.AA_andEachClassAccuracy(predicts,data.test_labels)
        for i in each_acc:
            print(i)
        print(average_acc)
        print(overall_acc)
        print(kappa)
        print()
        # print('ua =')
        for i in precision:
            print(i)
        print(average_precision)
        print(" Time: %.5f" % (time.time() - start_time))
        metrics = np.hstack((each_acc, average_acc.reshape(1, ), overall_acc.reshape(1, ), kappa.reshape(1, ), precision,
                            average_precision.reshape(1, )))
        np.savetxt('records/pavia'+'_'+repr(int(overall_acc * 10000))+'_'+repr(int(average_acc * 10000))+'_s'+repr(star_epoch)+'_'+repr(epoch)+'.txt', metrics.astype(str), fmt='%s', delimiter="\t", newline='\n')

        gt_flatten = data.gt.reshape(np.prod(data.gt.shape[:2]),)
        gt_NEW = gt_flatten.copy()
        gt_NEW[data.test_indices] = predicts + 1
        classification_map = np.reshape(gt_NEW, (data.gt.shape[0], data.gt.shape[1]))
        sio.savemat('figure/UP/HyperDGC_UP' + '_' + repr(int(overall_acc * 10000)) + '.mat',
                    {'classification_map': classification_map})
        hsi_pic = np.zeros((classification_map.shape[0], classification_map.shape[1], 3))
        for i in range(classification_map.shape[0]):
            for j in range(classification_map.shape[1]):
                if classification_map[i][j] == 0:
                    hsi_pic[i, j, :] = [0, 0, 0]
                if classification_map[i][j] == 1:
                    hsi_pic[i, j, :] = [216, 191, 216]
                if classification_map[i][j] == 2:
                    hsi_pic[i, j, :] = [0, 255, 0]
                if classification_map[i][j] == 3:
                    hsi_pic[i, j, :] = [0, 255, 255]
                if classification_map[i][j] == 4:
                    hsi_pic[i, j, :] = [45, 138, 86]
                if classification_map[i][j] == 5:
                    hsi_pic[i, j, :] = [255, 0, 255]
                if classification_map[i][j] == 6:
                    hsi_pic[i, j, :] = [255, 165, 0]
                if classification_map[i][j] == 7:
                    hsi_pic[i, j, :] = [159, 31, 239]
                if classification_map[i][j] == 8:
                    hsi_pic[i, j, :] = [255, 0, 0]
                if classification_map[i][j] == 9:
                    hsi_pic[i, j, :] = [255, 255, 0]
        utils.save_classification_map(hsi_pic / 255, classification_map, 24,
                                                        'figure/UP/HyperDGC_UP' + '_' + repr(int(overall_acc* 10000)) + '.png')
