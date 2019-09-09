import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import RandomSampler
import time
import copy
from mre.plotting import hv_dl_vis
from robust_loss_pytorch import adaptive
import warnings
from datetime import datetime
from tqdm import tqdm_notebook
from tensorboardX import SummaryWriter


# need data to be ordered thusly:
# image_sequence,width,hight,depth


class ChaosDataset(Dataset):
    def __init__(self, xr_ds, set_type='train', transform=None, clip=False, seed=100, test='01',
                 aug=True, sequence_mode='random', resize=False):

        xr_ds_test = xr_ds.sel(subject=[test])
        xr_ds = xr_ds.drop(test, dim='subject')
        np.random.seed(seed)
        shuffle_list = np.asarray(xr_ds.subject)
        np.random.shuffle(shuffle_list)

        if set_type == 'test':
            input_set = list(test)
        elif set_type == 'val':
            # input_set = xr_ds.subject_2d[2:20]
            input_set = list(shuffle_list[0:4])
        elif set_type == 'train':
            # input_set = xr_ds.subject_2d[:2]
            input_set = list(shuffle_list[4:])
        else:
            raise AttributeError('Must choose one of ["train", "val", "test"] for `set_type`.')

        # pick correct input set
        if set_type == 'test':
            xr_ds = xr_ds_test
        else:
            xr_ds = xr_ds.sel(subject=input_set)

        self.all_sequences = xr_ds.sequence.values
        if sequence_mode == 'random':
            self.my_sequence = [np.random.choice(self.all_sequences)]
        else:
            raise NotImplementedError('Only sequence_mode "random" is implemented')

        self.input_images = xr_ds.sel(sequence=self.my_sequence)['image'].transpose(
            'subject', 'sequence', 'z', 'y', 'x').values
        self.input_images = self.input_images.astype(np.float32)
        self.target_images = xr_ds.sel(sequence=self.my_sequence)['mask'].transpose(
            'subject', 'sequence', 'z', 'y', 'x').values
        self.target_images = self.target_images.astype(np.int32)

        self.transform = transform
        self.aug = aug
        self.clip = clip
        self.names = xr_ds.subject.values
        self.resize = resize

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        target = self.target_images[idx]
        if self.clip:
            if 't1' in self.my_sequence[0]:
                image = np.where(image >= 1500, 1500, image)
            else:
                image = np.where(image >= 2000, 2000, image)
            target = np.where(target > 0, 1, 0).astype(np.int32)

        if self.transform:
            if self.aug:
                rot_angle = np.random.uniform(-4, 4, 1)
                translations = np.random.uniform(-5, 5, 2)
                scale = np.random.uniform(0.95, 1.05, 1)
                restack = np.random.randint(-5, 5)
            else:
                rot_angle = 0
                translations = (0, 0)
                scale = 1
                restack = 0
            image = self.input_transform(image, rot_angle, translations, scale, restack)
            target = self.affine_transform(target, rot_angle, translations, scale, restack)

        image = torch.Tensor(image)
        target = torch.Tensor(target)

        return [image, target, self.names[idx]]

    def input_transform(self, input_image, rot_angle=0, translations=0, scale=1, restack=0):
        # normalize and offset image
        image = input_image
        image = np.where(input_image <= 1e-9, np.nan, input_image)
        mean = np.nanmean(image)
        std = np.nanstd(image)
        image = ((image - mean)/std) + 4
        # image = ((image - mean)/std)
        image = np.where(image != image, 0, image)

        # perform affine transfomrations
        image = self.affine_transform(image, rot_angle, translations, scale, restack)
        return image

    def affine_transform(self, image, rot_angle=0, translations=0, scale=1, restack=0):
        output_image = image.copy()
        for i in range(output_image.shape[0]):
            if (i+restack < 0) or (i+restack > output_image.shape[0]):
                output_image[0, i] = np.zeros_like(output_image[0, 0])
            else:
                output_image[0, i] = self.affine_transform_slice(output_image[0, i+restack],
                                                                 rot_angle, translations, scale)
        return output_image

    def affine_transform_slice(self, input_slice, rot_angle=0, translations=0, scale=1):
        input_slice = transforms.ToPILImage()(input_slice)
        input_slice = TF.affine(input_slice, angle=rot_angle,
                                translate=list(translations), scale=scale, shear=0)
        input_slice = transforms.ToTensor()(input_slice)
        return input_slice
