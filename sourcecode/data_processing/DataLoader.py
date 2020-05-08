""" Module for the data loading pipeline for the model to train """

import os
import numpy as np
from torch.utils.data import Dataset
import torch
from .new_transforms import *
import cv2

class FlatDirectoryImageDataset(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory images dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-2] == 'pt':
            im_arr = torch.load(img_name)
        else:
            im_arr = cv2.imread(img_name)
        if im_arr is None:
            im_arr = torch.load('ptf/167.pt')
        if im_arr.shape[-1] == 4:
            im_arr = im_arr[:,:,:3]

        if im_arr.shape[-1] != 3:
            print(im_arr.shape)
            im_arr = torch.load('ptf/167.pt')
        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(im_arr)

        # return the image:
        return img


class FoldersDistributedDataset(Dataset):
    """ pyTorch Dataset wrapper for the MNIST dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """

        dir_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for dir_name in dir_names:
            file_path = os.path.join(self.data_dir, dir_name)
            file_names = os.listdir(file_path)
            for file_name in file_names:
                possible_file = os.path.join(file_path, file_name)
                if os.path.isfile(possible_file):
                    files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] == 4:
            # ignore the alpha channel
            # in the image if it exists
            img = img[:3, :, :]

        # return the image:
        return img


def get_transform(new_size=None, flip_horizontal=False):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :param flip_horizontal: Whether to randomly mirror input images during training
    :return: image_transform => transform object from TorchVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize, \
        RandomHorizontalFlip

    image_transform = Compose([
        scale,
        resize_hard(256),
        hflip(),
        # gamma,
        # brightness,
        # contrast,
        scale,
        tofloatten,
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return image_transform


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dl
