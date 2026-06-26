from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *
import os, random, torch, numpy as np
from os import listdir
from os.path import join
from data.util import *

class PairedLowHighDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, red_aug=False):
        super(PairedLowHighDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.red_aug = red_aug
        self.low_files = sorted([join(data_dir+'/low', x) for x in listdir(data_dir+'/low') if is_image_file(x)])

    def __getitem__(self, index):
        im1 = load_img(self.low_files[index])
        high_file = self.low_files[index].replace('/low/', '/high/')
        im2 = load_img(high_file)
        _, file1 = os.path.split(self.low_files[index])
        _, file2 = os.path.split(high_file)
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        # red-decay augmentation (input only, R channel)
        if self.red_aug and random.random() < 0.5:
            g = random.uniform(0.3, 0.8)
            im1[0] = (im1[0] * g).clamp(0, 1)
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.low_files)

UIEBDatasetFromFolder = PairedLowHighDataset  # backward compat

def transform_uieb(size=256):
    from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop
    return Compose([
        RandomResizedCrop((size, size), scale=(0.5, 1.0)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def get_uieb_training_set(data_dir, size, red_aug=False):
    return PairedLowHighDataset(data_dir, transform=transform_uieb(size), red_aug=red_aug)

def get_euvp_training_set(data_dir, size, red_aug=False):
    return PairedLowHighDataset(data_dir, transform=transform_uieb(size), red_aug=red_aug)

def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform2():
    return Compose([ToTensor()])



def get_lol_training_set(data_dir,size):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir,size):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir,size):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())