import os
import torch
from torch.utils.data import Dataset
import numpy as np


TRAIN_MEAN = [20819.401209806543, 20501.830631237477, 19446.635075756352, 19837.531170690054]
              # 22503.501884778347, 13887.350799128100, 12093.980836390381, 6528.3341613259910]
TRAINVAL_MEAN = [20353.160445146474, 20117.942488387320, 19487.422707478603, 20181.519757616494]
                 # 22313.182497518817, 14895.992698284848, 13387.094821527728, 7290.1729489305080]
TEST_MEAN = [19343.974844417130, 18986.998935789370, 18306.343530577404, 18863.640906381603]
             # 22116.194165248537, 15886.431388183935, 13317.087609395947, 5851.3047235342220]


class L8_Biome_cls_WEAK(Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'pseudoMask/swinB_MFR')

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
        assert (len(self.images) == len(self.masks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        name = self.images[index]

        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        # rsData = rsData.transpose(2, 0, 1)

        if self.image_set.split('_')[0] == 'train':
            # print('train')
            mean = torch.tensor(TRAIN_MEAN)
        elif self.image_set.split('_')[0] =='trainval':
            # print('trainval')
            mean = torch.tensor(TRAINVAL_MEAN)
        elif self.image_set.split('_')[0] == 'test':
            # print('test')
            mean = torch.tensor(TEST_MEAN)

        img = torch.tensor(rsData[:4, :320, :320])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        target = np.load(self.masks[index])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target[:, :320, :320])

        return (img, target, name)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

class L8_Biome_cls_FULL(Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages_forVIS')
        mask_dir = os.path.join(voc_root, 'SegmentationClass_forVIS')

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
        assert (len(self.images) == len(self.masks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        name = self.images[index]

        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        # rsData = rsData.transpose(2, 0, 1)

        if self.image_set.split('_')[0] == 'train':
            # print('train')
            mean = torch.tensor(TRAIN_MEAN)
        elif self.image_set.split('_')[0] =='trainval':
            # print('trainval')
            mean = torch.tensor(TRAINVAL_MEAN)
        elif self.image_set.split('_')[0] == 'test':
            # print('test')
            mean = torch.tensor(TEST_MEAN)

        img = torch.tensor(rsData[:4, :320, :320])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        mask = np.load(self.masks[index])
        target = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)
        target[mask >= 192] = 1
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target[:, :320, :320])

        return (img, target, name)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)







