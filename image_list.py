from torchvision.datasets import VisionDataset
import warnings
import torch
from PIL import Image
import os
import os.path
import numpy as np
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageList(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root=None, transform=None, target_transform=None,  empty=False):
        super(ImageList, self).__init__(root, transform=transform, target_transform=target_transform)

        self.empty = empty
        if empty:
            self.samples = np.empty((1, 2), dtype='<U1000')
        else:
            self.samples = np.loadtxt(root, dtype=np.dtype((np.unicode_, 1000)), delimiter=' ')
        self.loader = pil_loader

    def __getitem__(self, index):

        path, label = self.samples[index]
        label = int(label)

        img0 = self.loader(path)

        if self.transform is not None:
            img = self.transform(img0)

        return img, label

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        if self.empty:
            self.samples = addition
            self.empty = False
        else:
            self.samples = np.concatenate((self.samples, addition), axis=0)
        return self.samples

    def remove_item(self, reduced):
        self.samples = np.delete(self.samples, reduced, axis=0)
        return self.samples