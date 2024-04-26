import os
import torch
from torch.utils.data import Dataset
import numpy as np


TRAIN_MEAN = [493.98764129, 478.56097384,424.43799190,520.79971255]
TRAINVAL_MEAN = [547.20599439,514.92978335,433.80227233,529.83716347]
TEST_MEAN = [483.67749218,483.75274581,483.81428944,483.88628532]

class GF1_cls_FULL(Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        block_dir = os.path.join(voc_root, 'block_label/bl_npy/')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        self.blocks = [os.path.join(block_dir, x + ".npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        name = self.images[index]

        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)


        if self.image_set.split('_')[0] == 'train':
            # print('train')
            mean = torch.tensor(TRAIN_MEAN)
        elif self.image_set.split('_')[0] =='trainval':
            # print('trainval')
            mean = torch.tensor(TRAINVAL_MEAN)
        elif self.image_set.split('_')[0] == 'test':
            # print('test')
            mean = torch.tensor(TEST_MEAN)

        img = torch.tensor(rsData[:,:320,:320])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        target = np.load(self.masks[index])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target[:, :320,:320])

        return (img, target, name)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

class GF1_cls_WEAK(Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')
        TOA_dir = os.path.join(voc_root, 'JPEGImages_TOA')

        mask_dir = os.path.join(voc_root, 'pseudoMask/MFC')

        block_dir = os.path.join(voc_root, 'block_label/bl_npy/')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.toas = [os.path.join(TOA_dir, x + ".npy") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        self.blocks = [os.path.join(block_dir, x + ".npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        name = self.images[index]

        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)

        if self.image_set.split('_')[0] == 'train':
            # print('train')
            mean = torch.tensor(TRAIN_MEAN)
        elif self.image_set.split('_')[0] =='trainval':
            # print('trainval')
            mean = torch.tensor(TRAINVAL_MEAN)
        elif self.image_set.split('_')[0] == 'test':
            # print('test')
            mean = torch.tensor(TEST_MEAN)

        img = torch.tensor(rsData[:,:320,:320])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re


        target = np.load(self.masks[index])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target[:, :320,:320])

        return (img, target, name)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)





