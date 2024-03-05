import argparse
import logging
import os
import subprocess
from io import StringIO
# import scipy.io as sio
import pandas as pd
# import tensorflow.compat.v1 as tf
import tensorflow as tf
# tf2.test.gpu_device_name
# tf.disable_v2_behavior()
import configs
import utils
from datahandler import DataHandler
from dgc import DGC

def main(tag, seed,train_rate, dataset, alpha, window,star_epoch,kernel,augrate):
    opts = getattr(configs, 'config_%s' % dataset)
    opts['work_dir'] = './results/%s/' % tag
    print(seed,alpha,window, kernel, star_epoch,augrate,train_rate)
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'],
                                  'checkpoints'))

    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    data = DataHandler(opts, window, seed,train_rate)
    model = DGC(opts, tag, alpha = alpha, window=window, kernel=kernel,augrate=augrate,cls_cnt=data.cls_cnt)
    model.train(data,seed,star_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='paviau',
                        help='dataset [mnist/celeba/paviau]')
    parser.add_argument("--seed", type=int, default=1,
                        help='random seed for imbalance data generation')
    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    os.environ["OMP_NUM_THREADS"] = "8"
    
    dataset_name = FLAGS.exp
    # seed = FLAGS.seed
    # tag = '%s_seed%02d' % (dataset_name, seed) ok:1330, 1220, 1336, 1337, 1226,1445,1339,1550
    # main(tag, seed, dataset_name) 1330,1220,1336, 1337,1226,1445,1339,1550,1440,     1330,1220,1336,1337,1256,
    # ITER = 10
    # seeds = [1330, 1220, 1336, 1337, 1226,1445,1339,1550,1256]
    # for index_iter in range(ITER):
    #     seed = seeds[index_iter]
    #     tag = '%s_seed%02d' % (dataset_name, seed)
    #     main(tag, seed, dataset_name)90,100,110,
    ITER = 5
    seeds = [1330, 1220,1256,1445,1440,1223,1440,1234,1226,1336]
    #windows = [7,9,11,13,15]
    star_epochs = [120]
    train_rates=[0.01]#0.015,0.02,
    alphas = [110] # 0.3,0.5,0.7,0.9,
    for index_iter in range(4):
        #window = windows[index_iter]
        train_rate = train_rates[index_iter]
        for i in range(10):
            seed = seeds[i]
            tag = '%s_seed%02d' % (dataset_name, seed)
            # for alpha in [30,40,60,70]:
            main(tag, seed, train_rate, dataset_name, alpha=120, window=13, star_epoch=130, kernel=32,augrate=1.2)
    # windows = [13,15]
    # kernels = [24,32,40,48]
    # augrates = [1.6, 1.4, 1.2, 0.8, 0.6, 0.4, 0.2, 0]
    # # for k in range(4):
    # #     kernel = kernels[k]
    # for i in range(3):
    #     # augrate = augrates[i]
    #     #window = windows[i]
    #     star_epoch = star_epochs[i]
    #     for k in range(1):
    #         alpha = alphas[k]
    #         for j in range(10):
    #             seed = seeds[j]
    #             tag = '%s_seed%02d' % (dataset_name, seed)
    #                 # for alpha in [30,40,60,70]:
    #             main(tag, seed, dataset_name, alpha=alpha, window=13, star_epoch=star_epoch, kernel=32,augrate=1.2)