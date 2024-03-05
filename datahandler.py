import math
import os
import numpy as np
import cv2
import scipy.io as sio
from dataset.dataset import load_data_HSI
from utils import shuffle
import glob
import h5py
  

datashapes = {
    'mnist': [28, 28, 1],
    'celeba': [64, 64, 3],
    'paviau': [13, 13, 103, 1],
}
w, h = 512, 512
epoch = 1
num = 1  # number of  pic to agumen
class DataHandler(object):
    def __init__(self, opts, window, seed,train_rate):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.test_data = None
        self.labels = None
        self.data_augx = None
        self.labels_augx=None
        self.test_labels = None
        self.train_dataset = None
        self.test_dataset = None
        self.class_counts = None
        self._load_data(opts, window, seed,train_rate)


    def _load_data(self, opts,window, seed,train_rate):
        if opts['dataset'].lower() in ('mnist', 'celeba','paviau'):
        #     (self.data, self.labels), (self.test_data, self.test_labels) = load_data(opts['dataset'], seed,
        #                                                                              imbalance=True)
            (self.data, self.labels), (self.test_data, self.test_labels),self.test_indices,self.gt,self.cls_cnt = load_data_HSI(opts['dataset'],seed, 103, window,train_rate,imbalance=True)
            # with h5py.File(os.path.join(opts['work_dir'],'test.h5'), 'w') as f:
            #     f.create_dataset('test_data',data=self.test_data,dtype='f')
            #     f.create_dataset('test_labels',data=self.test_labels)
            #     f.create_dataset('test_indices',data=self.test_indices)
            #     f.create_dataset('gt',data=self.gt)                
            #                                                             
            # hdf5storage.savemat(opts['work_dir'],{'data':[self.test_data], 'label':[self.test_labels]} ,do_compression=True) 
            # self.data = self.radiation_noise(self.data)
            # self.data, self.labels = self.flip_augmentation(self.data,self.labels)
            # self.data, self.labels = self.add_noises(self.data, self.labels)
            if 'augment_x' in opts and opts['augment_x']:
                self.data_augx, self.labels_augx = self.oversampling(opts, self.data, self.labels, seed)
            self.num_points = len(self.data)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

        self.class_counts = [np.count_nonzero(self.labels == c) for c in range(opts['n_classes'])]
        print("[ statistic ]")
        print("Total train: ", self.num_points)
        print(self.class_counts)
        print("Total test: ", len(self.test_labels))
        print([np.count_nonzero(self.test_labels == c) for c in range(opts['n_classes'])])

    # def oversampling(self, opts, x, y, seed):
    #     n_classes = opts['n_classes']
    #     class_cnt = [np.count_nonzero(y == c) for c in range(n_classes)]
    #     max_class_cnt = max(class_cnt)
    #     x_aug_list = []
    #     y_aug_list = []
    #     x_aug_list_noise = []
    #     y_aug_list_noise = []
    #     x_noise = []
    #     aug_rate = opts['aug_rate']
    #     if aug_rate <= 0:
    #         return x, y
    #     aug_nums = [aug_rate * (max_class_cnt - class_cnt[i]) for i in range(n_classes)]
    #     rep_nums = [aug_num / class_cnt[i] - class_cnt[i] for i, aug_num in enumerate(aug_nums)]
    #     for i in range(n_classes):
    #         idx = (y == i)
    #         if rep_nums[i] <= 0.:
    #             x_aug_list.append(x[idx])
    #             y_aug_list.append(y[idx])
    #             continue
    #         n_c = np.count_nonzero(idx)
    #         if n_c == 0:
    #             continue
    #         x_aug_list.append(x[idx])
    #         y_aug_list.append(y[idx])
    #
    #         x_aug_list_noise.append(x[idx])
    #         x_aug__noise = np.array(x_aug_list_noise)
    #         y_aug_list_noise.append(y[idx])
    #         y_aug__noise = np.array(y_aug_list_noise)
    #         x_aug__noise = self.radiation_noise(x_aug__noise[0,:,:,:,:,:])
    #         x_noise,y_noise = self.flip_augmentation(x_aug__noise,y_aug__noise)
    #         x_aug_list.append(np.repeat(x_noise[idx], repeats=math.ceil(rep_nums[i]), axis=0)[:math.floor(n_c * (rep_nums[i]))])
    #         y_aug_list.append(np.repeat(y_noise[idx], repeats=math.ceil(rep_nums[i]), axis=0)[:math.floor(n_c * (rep_nums[i]))])
    #     if len(x_aug_list) == 0:
    #         return x, y
    #     x_aug = np.concatenate(x_aug_list, axis=0)
    #     y_aug = np.concatenate(y_aug_list, axis=0)
    #     x_aug, y_aug = shuffle(x_aug, y_aug, seed)
    #     print([np.count_nonzero(y_aug == c) for c in range(n_classes)])
    #     return x_aug, y_aug
    #
    # def radiation_noise(self, data, alpha_range=(0.9, 1.1), beta=1/25):
    #     alpha = np.random.uniform(*alpha_range)
    #     noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    #     return alpha * data + beta * noise
    #
    # def flip_augmentation(self,data,label): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    #     horizontal = np.random.random() > 0.5 # True
    #     vertical = np.random.random() > 0.5 # False
    #     if horizontal:
    #         data = np.fliplr(data)
    #         label = np.fliplr(label)
    #     if vertical:
    #         data = np.flipud(data)
    #         label = np.flipud(label)
    #     return data,label
    #
    def oversampling(self, opts, x, y, seed):
         n_classes = opts['n_classes']
         class_cnt = [np.count_nonzero(y == c) for c in range(n_classes)]
         max_class_cnt = max(class_cnt)
         x_aug_list = []
         y_aug_list = []
         aug_rate = opts['aug_rate']
         if aug_rate <= 0:
             return x, y
         aug_nums = [aug_rate * (max_class_cnt - class_cnt[i]) for i in range(n_classes)]
         rep_nums = [aug_num / class_cnt[i] for i, aug_num in enumerate(aug_nums)]
         for i in range(n_classes):
             idx = (y == i)
             if rep_nums[i] <= 0.:
                 x_aug_list.append(x[idx])
                 y_aug_list.append(y[idx])
                 continue
             n_c = np.count_nonzero(idx)
             if n_c == 0:
                 continue
             x_aug_list.append(
                 np.repeat(x[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
             y_aug_list.append(
                 np.repeat(y[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
         if len(x_aug_list) == 0:
             return x, y
         x_aug = np.concatenate(x_aug_list, axis=0)
         y_aug = np.concatenate(y_aug_list, axis=0)
         x_aug, y_aug = shuffle(x_aug, y_aug, seed)
         print([np.count_nonzero(y_aug == c) for c in range(n_classes)])
         return x_aug, y_aug