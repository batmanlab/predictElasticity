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
from robust_loss_pytorch import adaptive
import warnings
from datetime import datetime
from tqdm import tqdm_notebook
from tensorboardX import SummaryWriter


# need data to be ordered thusly:
# image_sequence,width,hight,depth


class ChaosDataset(Dataset):
    def __init__(self, xr_ds, set_type='train', transform=None, clip=False, seed=100, test='01',
                 aug=True, sequence_mode='random', resize=False, split=0.2, model_arch='3D',
                 verbose=False):

        self.verbose = verbose
        self.model_arch = model_arch
        if type(test) is not list:
            test = [test]
        xr_ds_test = xr_ds.sel(subject=test)
        xr_ds = xr_ds.drop(test, dim='subject')
        np.random.seed(seed)
        shuffle_list = np.asarray(xr_ds.subject)
        np.random.shuffle(shuffle_list)

        if set_type == 'test':
            input_set = test
        elif set_type == 'val':
            # input_set = xr_ds.subject_2d[2:20]
            input_set = list(shuffle_list[0:3])
        elif set_type == 'train':
            # input_set = xr_ds.subject_2d[:2]
            input_set = list(shuffle_list[3:])
        else:
            raise AttributeError('Must choose one of ["train", "val", "test"] for `set_type`.')

        # pick correct input set, remove test subjects
        if set_type == 'test':
            xr_ds = xr_ds_test
        else:
            xr_ds = xr_ds.sel(subject=input_set)

        # pick correct sequences
        self.all_sequences = xr_ds.sequence.values
        if sequence_mode == 'random':
            self.my_sequence = [np.random.choice(self.all_sequences)]
        elif sequence_mode == 'all':
            self.my_sequence = ['t1_in', 't1_out', 't2']
        else:
            self.my_sequence = [sequence_mode]

        # assign input and target elements, based on 2d or 3d arch
        if self.model_arch == '2D':
            xr_ds = xr_ds.stack(subject_2d=('subject', 'z')).reset_index('subject_2d')
            subj_2d_coords = [f'{i.subject.values}_{i.z.values}' for i in xr_ds.subject_2d]
            xr_ds = xr_ds.assign_coords(subject_2d=subj_2d_coords)
            bad_slices = []
            for i in subj_2d_coords:
                if xr_ds.sel(subject_2d=i).mask.sum() < 10:
                    bad_slices.append(i)
            xr_ds = xr_ds.drop(bad_slices, dim='subject_2d')

            self.names = xr_ds.subject_2d.values
            self.input_images = xr_ds.sel(sequence=self.my_sequence)['image'].transpose(
                'subject_2d', 'sequence', 'y', 'x').values
            self.input_images = self.input_images.astype(np.float32)
            self.target_images = xr_ds.sel(sequence=self.my_sequence)['mask'].transpose(
                'subject_2d', 'sequence', 'y', 'x').values
            self.target_images = self.target_images.astype(np.int32)

        else:
            self.names = xr_ds.subject.values
            self.input_images = xr_ds.sel(sequence=self.my_sequence)['image'].transpose(
                'subject', 'sequence', 'z', 'y', 'x').values
            self.input_images = self.input_images.astype(np.float32)
            self.target_images = xr_ds.sel(sequence=self.my_sequence)['mask'].transpose(
                'subject', 'sequence', 'z', 'y', 'x').values
            self.target_images = self.target_images.astype(np.int32)

        # Additional flags
        self.transform = transform
        self.aug = aug
        self.clip = clip
        self.resize = resize

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):

        if self.verbose:
            print(self.names[idx])
        if self.model_arch == '3D':
            image, target = self.get_data_aug_3d(idx)
        elif self.model_arch == '2D':
            image, target = self.get_data_aug_2d(idx)

        image = torch.Tensor(image)
        target = torch.Tensor(target)

        return [image, target, self.names[idx]]

    def get_data_aug_3d(self, idx):
        '''get data (image and target), and apply augmentations if indicated'''
        image = self.input_images[idx]
        target = self.target_images[idx]

        if self.clip:
            for i, seq in enumerate(self.my_sequence):
                if 't1' in seq:
                    image[i, :] = np.where(image[i, :] >= 1500, 1500, image[i, :])
                else:
                    image[i, :] = np.where(image[i, :] >= 2000, 2000, image[i, :])
            target = np.where(target > 0, 1, 0).astype(np.int32)

        if self.transform:
            if self.aug:
                rot_angle = np.random.uniform(-2, 2, 1)
                translations = np.random.uniform(-2, 2, 2)
                scale = np.random.uniform(0.95, 1.05, 1)
                # restack = np.random.randint(-3, 4)
                restack = 0
                # restack = np.random.randint(-6, -1)
                # restack = 0
                # flip = np.random.randint(0, 2)
            else:
                rot_angle = 0
                translations = (0, 0)
                scale = 1
                restack = 0
            for i in range(len(self.my_sequence)):
                image[i, :] = self.input_transform_3d(image[i:i+1, :], rot_angle,
                                                      translations, scale, restack)
                target[i, :] = self.affine_transform_3d(target[i:i+1, :], rot_angle,
                                                        translations, scale, restack)
        return image, target

    def get_data_aug_2d(self, idx):
        '''get data (image and target), and apply augmentations if indicated'''
        image = self.input_images[idx]
        target = self.target_images[idx]

        if self.clip:
            for i, seq in enumerate(self.my_sequence):
                if 't1' in seq:
                    image[i, :] = np.where(image[i, :] >= 1500, 1500, image[i, :])
                else:
                    image[i, :] = np.where(image[i, :] >= 2000, 2000, image[i, :])
            target = np.where(target > 0, 1, 0).astype(np.int32)

        if self.transform:
            if self.aug:
                rot_angle = np.random.uniform(-2, 2, 1)
                translations = np.random.uniform(-2, 2, 2)
                scale = np.random.uniform(0.95, 1.05, 1)
            else:
                rot_angle = 0
                translations = (0, 0)
                scale = 1
            for i in range(len(self.my_sequence)):
                image[i, :] = self.input_transform_2d(image[i, :], rot_angle,
                                                      translations, scale)
                target[i, :] = self.affine_transform_2d(target[i, :], rot_angle,
                                                        translations, scale)
        return image, target

    def input_transform_3d(self, input_image, rot_angle=0, translations=0, scale=1, restack=0):
        # normalize and offset image
        image = input_image
        # image = np.where(input_image <= 1e-9, np.nan, input_image)
        mean = np.nanmean(image)
        std = np.nanstd(image)
        # image = ((image - mean)/std) + 4
        image = ((image - mean)/std)
        image = np.where(image != image, 0, image)

        # perform affine transfomrations
        image = self.affine_transform_3d(image, rot_angle, translations, scale, restack)
        return image

    def input_transform_2d(self, input_image, rot_angle=0, translations=0, scale=1, restack=0):
        # normalize and offset image
        image = input_image
        # image = np.where(input_image <= 1e-9, np.nan, input_image)
        mean = np.nanmean(image)
        std = np.nanstd(image)
        # image = ((image - mean)/std) + 4
        image = ((image - mean)/std)
        image = np.where(image != image, 0, image)

        # perform affine transfomrations
        image = self.affine_transform_2d(image, rot_angle, translations, scale)
        return image

    def affine_transform_3d(self, image, rot_angle=0, translations=0, scale=1, restack=0):
        output_image = image.copy()
        if self.verbose:
            print(f'rot_angle: {rot_angle}, translations: {translations},'
                  f'scale: {scale}, restack={restack}')

        # for i in range(output_image.shape[0]):
        #     if (i+restack < 0) or (i+restack > output_image.shape[0]):
        #         output_image[0, i] = np.zeros_like(output_image[0, 0])
        #     else:
        #         output_image[0, i] = self.affine_transform_slice(output_image[0, i+restack],
        #                                                          rot_angle, translations, scale)

        for i in range(output_image.shape[1]):
            if (i+restack < 0) or (i+restack > output_image.shape[1]-1):
                output_image[0, i] = np.zeros_like(output_image[0, 0])
            else:
                output_image[0, i] = self.affine_transform_2d(image[0, i+restack],
                                                              rot_angle, translations, scale)
        return output_image

    def affine_transform_2d(self, input_slice, rot_angle=0, translations=0, scale=1):
        input_slice = transforms.ToPILImage()(input_slice)
        input_slice = TF.affine(input_slice, angle=rot_angle,
                                translate=list(translations), scale=scale, shear=0)
        input_slice = transforms.ToTensor()(input_slice)
        return input_slice
