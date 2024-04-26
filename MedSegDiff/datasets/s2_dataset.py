import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

# TRAIN_MEAN = [0.38018656,0.40501326,0.41393024,0.451276]
# TRAINVAL_MEAN = [0.44421858,0.48062754,0.4844972,0.50280094]
# TEST_MEAN = [0.33277953,0.34559596,0.35245013,0.4017625]

TRAIN_MEAN = [0.35086408,0.32679248,0.2967656,0.3166021,0.32596204,0.3476482,
              0.3623534,0.34722617,0.36966884,0.17354482,0.025884438,0.23780519,0.18993858]
TEST_MEAN = [0.29571676,0.2690076,0.24087743,0.24140006,0.24947266,0.2841256,
             0.3054672,0.2933387,0.3183856,0.13435796,0.0288091,0.20810705,0.14918126]



class sentinel_cls_FULL(Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')

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
        rsData = rsData.transpose(2, 0, 1)

        if self.image_set.split('_')[0] == 'train':
            # print('train')
            mean = torch.tensor(TRAIN_MEAN)
        elif self.image_set.split('_')[0] == 'test':
            # print('test')
            mean = torch.tensor(TEST_MEAN)


        img = torch.tensor(rsData)
        mean_re = mean.view(13, 1, 1).expand((13, 320, 320))
        img = img - mean_re
        # img = F.interpolate(img.unsqueeze(dim=0), size=(160, 160), mode='nearest').squeeze(dim=0)

        target = np.load(self.masks[index])
        target = np.expand_dims(target, axis=0)
        target = torch.tensor(target, dtype = torch.float32)
        # target = F.interpolate(target.unsqueeze(dim=0), size=(160, 160), mode='nearest').squeeze(dim=0)

        return (img, target, name)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

class sentinel_cls_WEAK(Dataset):
    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        # mask_dir = os.path.join(voc_root, 'pseudoMask/swinB_bbone')
        mask_dir = '/home/visint-book/sentinel_dst/data/PseudoMask/swinB_bbone/'


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
        rsData = rsData.transpose(2, 0, 1)

        if self.image_set.split('_')[0] == 'train':
            # print('train')
            mean = torch.tensor(TRAIN_MEAN)
        elif self.image_set.split('_')[0] == 'test':
            # print('test')
            mean = torch.tensor(TEST_MEAN)

        # img = torch.tensor(np.asarray([rsData[1, :320, :320],rsData[2, :320, :320],rsData[3, :320, :320],rsData[7, :320, :320]]))
        # mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        # img = img - mean_re

        img = torch.tensor(rsData)
        mean_re = mean.view(13, 1, 1).expand((13, 320, 320))
        img = img - mean_re
        # img = F.interpolate(img.unsqueeze(dim=0),size=(160, 160),mode='nearest').squeeze(dim=0)


        target = np.load(self.masks[index])
        target = np.expand_dims(target, axis=0)
        target = torch.tensor(target, dtype = torch.float32)
        # target = F.interpolate(target.unsqueeze(dim=0),size=(160, 160),mode='nearest').squeeze(dim=0)


        return (img, target, name)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)




