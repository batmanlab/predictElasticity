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
    def __init__(self, xa_ds, set_type='train', transform=None, clip=False, seed=100, test='01',
                 aug=True, sequence='random', resize=False):
        inputs = ['t1_in', 't1_out', 't2']

        xa_ds_test = xa_ds.sel(subject=[test])
        xa_ds = xa_ds.drop(test, dim='subject')
        np.random.seed(seed)
        shuffle_list = np.asarray(xa_ds.subject)
        np.random.shuffle(shuffle_list)

        if set_type == 'test':
            input_set = list(test)
        elif set_type == 'val':
            # input_set = xa_ds.subject_2d[2:20]
            input_set = list(shuffle_list[0:8])
        elif set_type == 'train':
            # input_set = xa_ds.subject_2d[:2]
            input_set = list(shuffle_list[8:])
        else:
            raise AttributeError('Must choose one of ["train", "val", "test"] for `set_type`.')

        # pick correct input set
        if set_type == 'test':
            xa_ds = xa_ds_test
        else:
            xa_ds = xa_ds.sel(subject=input_set)

        self.input_images = xa_ds.sel(sequence=inputs)['image'].transpose(
            'subject', 'sequence', 'z', 'y', 'x').image.values
        self.target_images = xa_ds.sel(sequence=inputs)['mask'].transpose(
            'subject', 'sequence', 'z', 'y', 'x').image.values

        self.transform = transform
        self.aug = aug
        self.clip = clip
        self.names = xa_ds.subject.values
        self.target_bins = target_bins
        self.resize = resize

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        target = self.target_images[idx]
        if self.clip:
            image[0, :, :, :]  = np.where(image[0, :, :, :] >= 1500, 1500, image[0, :, :, :])
            image[1, :, :, :]  = np.where(image[1, :, :, :] >= 1500, 1500, image[1, :, :, :])
            image[2, :, :, :]  = np.where(image[2, :, :, :] >= 2000, 2000, image[2, :, :, :])
            target = np.where(target > 0, 1, 0)

        if self.transform:
            if self.aug:
                rot_angle = np.random.uniform(-4, 4, 1)
                translations = np.random.uniform(-5, 5, 2)
                scale = np.random.uniform(0.95, 1.05, 1)
            else:
                rot_angle = 0
                translations = (0, 0)
                scale = 1
            image = self.input_transform(image, rot_angle, translations, scale)
            target = self.affine_transform(target, rot_angle, translations, scale)

        if self.resize:
            image_0 = transforms.ToPILImage()(image.numpy()[0])
            image_0 = transforms.functional.resize(image_0, (224, 224))
            image_0 = transforms.ToTensor()(image_0)
            image_1 = transforms.ToPILImage()(image.numpy()[1])
            image_1 = transforms.functional.resize(image_1, (224, 224))
            image_1 = transforms.ToTensor()(image_1)
            image_2 = transforms.ToPILImage()(image.numpy()[2])
            image_2 = transforms.functional.resize(image_2, (224, 224))
            image_2 = transforms.ToTensor()(image_2)
            image = torch.cat((image_0, image_1, image_2))
            target = transforms.ToPILImage()(target.numpy()[0])
            target = transforms.Resize((224, 224))(target)
            target = transforms.ToTensor()(target)
            mask = transforms.ToPILImage()(mask.numpy()[0])
            mask = transforms.Resize((224, 224))(mask)
            mask = transforms.ToTensor()(mask)

        image = torch.Tensor(image)
        target = torch.Tensor(target)
        mask = torch.Tensor(mask)

        return [image, target, mask, self.names[idx]]

    def affine_transform(self, input_slice, rot_angle=0, translations=0, scale=1):
        input_slice = transforms.ToPILImage()(input_slice)
        input_slice = TF.affine(input_slice, angle=rot_angle,
                                translate=list(translations), scale=scale, shear=0)
        input_slice = transforms.ToTensor()(input_slice)
        return input_slice

    def input_transform(self, input_image, rot_angle=0, translations=0, scale=1):

        # normalize and offset image
        image = input_image
        image = np.where(input_image <= 1e-9, np.nan, input_image)
        mean = np.nanmean(image, axis=(1, 2))
        std = np.nanstd(image, axis=(1, 2))
        image = ((image.T - mean)/std).T + 4
        image = np.where(image != image, 0, image)

        # perform affine transfomrations
        image0 = self.affine_transform(image[0], rot_angle, translations, scale)
        image1 = self.affine_transform(image[1], rot_angle, translations, scale)
        image2 = self.affine_transform(image[2], rot_angle, translations, scale)
        return torch.cat((image0, image1, image2))

