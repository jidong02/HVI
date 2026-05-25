import os
import random
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
from data.util import *


class UIEBDatasetFromFolder(data.Dataset):
    """
    UIEB 训练集
    目录结构:
        data_dir/
        ├── low/    # 水下退化图（输入）
        └── high/   # 参考图（GT）
    """
    def __init__(self, data_dir, transform=None, min_size=288):
        super(UIEBDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.min_size = min_size  # 略大于 cropSize=256，确保有裁剪余地

        folder = join(data_dir, 'low')
        folder2 = join(data_dir, 'high')
        self.data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        self.data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])
        assert len(self.data_filenames) == len(self.data_filenames2), \
            f"low/ ({len(self.data_filenames)}) 和 high/ ({len(self.data_filenames2)}) 数量不一致"
        print(f"[UIEB] Loaded {len(self.data_filenames)} training pairs from {data_dir}")

    def _ensure_min_size(self, im1, im2):
        """如果图的短边 < min_size，等比例放大；low 和 high 同步放大以保配对"""
        w, h = im1.size
        if min(w, h) < self.min_size:
            scale = self.min_size / min(w, h)
            new_w = int(w * scale + 0.5)
            new_h = int(h * scale + 0.5)
            im1 = im1.resize((new_w, new_h), Image.BICUBIC)
            im2 = im2.resize((new_w, new_h), Image.BICUBIC)
        return im1, im2

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])

        # 关键：小图先放大
        im1, im2 = self._ensure_min_size(im1, im2)

        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed); torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed); torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.data_filenames)
